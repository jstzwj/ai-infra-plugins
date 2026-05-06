# Apache TVM Reference - Chapter 23: Dataflow Pattern Language (DPL)

This reference covers the Dataflow Pattern Language (DPL) in Apache TVM's Relax frontend. DPL is a declarative pattern matching and rewriting facility that enables graph-level transformations on Relax IR. It is the primary mechanism for operator fusion, backend dispatch, and custom graph rewrites in the Relax compilation pipeline.

---

## 23.1 Overview

### 23.1.1 What is DPL?

The Dataflow Pattern Language (DPL) is a domain-specific language embedded in Python that allows developers to describe sub-graph patterns in Relax IR and either match them or rewrite them. DPL patterns describe the structural shape of computation graphs -- the operators used, their connectivity, and optional constraints on types, shapes, and attributes.

### 23.1.2 Why DPL?

Traditional graph rewriting requires manually traversing IR nodes, matching against expected structures, and constructing replacement sub-graphs. This process is error-prone, verbose, and hard to compose. DPL provides:

- **Declarative specification**: Describe what to match, not how to traverse.
- **Composability**: Patterns can be combined with logical operators (AND, OR, NOT) and chained with sequence operators.
- **Type safety**: Patterns can specify type and shape constraints that are checked during matching.
- **Integration with passes**: DPL patterns integrate directly with `FuseOpsByPattern`, `rewrite_call`, and `rewrite_bindings`.

### 23.1.3 Three-Step Workflow

Every DPL-based transformation follows three steps:

1. **Build**: Construct a pattern object that describes the sub-graph shape to search for.
2. **Match**: Apply the pattern against Relax IR to find matching sub-graphs.
3. **Rewrite**: Optionally replace matched sub-graphs with new structures.

```python
from tvm.relax.dpl import *

# Step 1: Build pattern
x = wildcard()
w = wildcard()
matmul = is_op("relax.matmul")(x, w)
bias = wildcard()
add = is_op("relax.add")(matmul, bias)

# Step 2 & 3: Match and rewrite
from tvm.relax.dpl import rewrite_call
def rewriter(matched_expr, matchings):
    return relax.op.call_pure_packed(
        "fused_matmul_add",
        matchings[x], matchings[w], matchings[bias],
        sinfo_args=relax.TensorStructInfo(matched_expr.struct_info),
    )

new_func = rewrite_call(add, rewriter, func)
```

---

## 23.2 Pattern Construction

DPL provides a rich set of primitives for constructing patterns. Patterns are first-class objects that can be stored, composed, and reused.

### 23.2.1 Wildcard Pattern

The wildcard pattern matches any expression regardless of its type, shape, or value.

```python
from tvm.relax.dpl import *

# Matches any single expression
pat = wildcard()

# Named wildcard -- useful for capturing matched expressions
x = wildcard()
y = wildcard()
```

**Usage**: Wildcards are the most basic building block. They are used as "holes" in patterns that will be filled by any expression during matching. Named wildcards serve as capture variables that can be referenced in the rewriter.

### 23.2.2 Operator Pattern

Matches a specific Relax operator call.

```python
# Match a specific operator
add_pat = is_op("relax.add")
matmul_pat = is_op("relax.matmul")
conv2d_pat = is_op("relax.nn.conv2d")
relu_pat = is_op("relax.nn.relu")
```

**Signature:**
```python
def is_op(op_name: str) -> DFPattern
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_name` | `str` | Fully qualified operator name (e.g., `"relax.add"`) |

**Return value:** A `DFPattern` that matches calls to the specified operator.

**Details:**

The operator name must match the registered name in the Relax operator registry. Common operators include:

| Category | Operators |
|----------|-----------|
| Arithmetic | `relax.add`, `relax.subtract`, `relax.multiply`, `relax.divide` |
| Comparison | `relax.equal`, `relax.greater`, `relax.less` |
| Linear algebra | `relax.matmul`, `relax.linear`, `relax.einsum` |
| Neural network | `relax.nn.conv2d`, `relax.nn.relu`, `relax.nn.gelu`, `relax.nn.softmax` |
| Reduction | `relax.sum`, `relax.max`, `relax.min`, `relax.mean` |
| Shape | `relax.reshape`, `relax.transpose`, `relax.permute_dims`, `relax.expand_dims` |
| Creation | `relax.full`, `relax.zeros`, `relax.ones`, `relax.arange` |

### 23.2.3 Constant Pattern

Matches a constant expression.

```python
# Match any constant
const_pat = is_const()

# Named constant capture
weight = is_const()
```

**Signature:**
```python
def is_const() -> DFPattern
```

### 23.2.4 Variable Patterns

Match specific kinds of variable nodes in Relax IR.

```python
# Match a Var (function parameter or let-bound variable)
var_pat = is_var("x")

# Match a DataflowVar (variable within a DataflowBlock)
dfv_pat = is_dfv("y")

# Match a GlobalVar (reference to another function in the module)
gv_pat = is_gv("my_func")

# Match without name constraint
any_var = is_var()
any_dfv = is_dfv()
any_gv = is_gv()
```

**Signatures:**
```python
def is_var(name: str = None) -> DFPattern
def is_dfv(name: str = None) -> DFPattern
def is_gv(name: str = None) -> DFPattern
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` or `None` | Optional name to match. If `None`, matches any variable of that type. |

### 23.2.5 Tuple Patterns

Match tuple construction and element extraction.

```python
# Match a tuple with specific field patterns
a, b = wildcard(), wildcard()
tup_pat = is_tuple([a, b])

# Match a tuple with any number of fields
any_tup = is_tuple(None)

# Match tuple element extraction
first = tup_pat[0]       # TupleGetItemPattern
second = tup_pat[1]

# Match TupleGetItem directly
tgi_pat = is_tuple_get_item(wildcard(), index=0)
```

**Signatures:**
```python
def is_tuple(fields: list = None) -> DFPattern
def is_tuple_get_item(tuple_pattern: DFPattern, index: int = None) -> DFPattern
```

### 23.2.6 Call Patterns

When a pattern is "called" with argument patterns, it creates a `CallPattern` that matches function calls with the given operator pattern and argument patterns.

```python
# Basic call pattern
x = wildcard()
y = wildcard()
add_pat = is_op("relax.add")(x, y)
# Matches: R.add(x_expr, y_expr) for any x_expr, y_expr

# Nested call pattern
inp = wildcard()
w = wildcard()
b = wildcard()
matmul = is_op("relax.matmul")(inp, w)
add = is_op("relax.add")(matmul, b)
# Matches: R.add(R.matmul(inp_expr, w_expr), b_expr)

# Unary operator pattern
relu = is_op("relax.nn.relu")(wildcard())

# Operator with no arguments (rare)
# some_pat = is_op("relax.some_op")()
```

**Variadic arguments:**

Some operators accept a variable number of arguments (e.g., `concatenate`). DPL supports matching variadic arguments:

```python
# Match concatenate with any number of arguments
concat = is_op("relax.concat")(wildcard(), varg_default_wildcard=True)

# The matched expression's arguments will be captured
```

### 23.2.7 Specialized Call Patterns

For operators that are not standard Relax operators but instead use special calling conventions:

```python
# Match R.call_tir(prim_func, args, out_sinfo)
call_tir_pat = is_call_tir("my_func", [wildcard(), wildcard()])

# Match R.call_dps_packed(packed_func, args, out_sinfo)
call_dps_pat = is_call_dps_packed("my_packed_func", [wildcard()])

# Match R.call_packed(packed_func, args, ...)
call_packed_pat = is_call_packed("my_packed_func", [wildcard(), wildcard()])
```

**Signatures:**
```python
def is_call_tir(func_name: str = None, args: list = None) -> DFPattern
def is_call_dps_packed(func_name: str = None, args: list = None) -> DFPattern
def is_call_packed(func_name: str = None, args: list = None) -> DFPattern
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `func_name` | `str` or `None` | Optional function name to match |
| `args` | `list` or `None` | Optional list of argument patterns |

---

## 23.3 Pattern Constraints

Constraints allow patterns to specify additional conditions beyond structural matching. A pattern with constraints will only match if all constraints are satisfied.

### 23.3.1 Type Constraint (has_dtype)

Restricts the match to expressions with a specific data type.

```python
# Match matmul that produces float16 output
fp16_matmul = is_op("relax.matmul")(wildcard(), wildcard()).has_dtype("float16")

# Match float32 add
fp32_add = is_op("relax.add")(wildcard(), wildcard()).has_dtype("float32")

# Common types: "float16", "float32", "float64", "int8", "int32", "bool"
```

**Signature:**
```python
pattern.has_dtype(dtype: str) -> DFPattern
```

### 23.3.2 Shape Constraint (has_shape)

Restricts the match to expressions with a specific tensor shape.

```python
# Match add with exactly (128, 128) shape
square_add = is_op("relax.add")(wildcard(), wildcard()).has_shape((128, 128))

# Match reshape to (1, 128)
reshape = is_op("relax.reshape")(wildcard()).has_shape((1, 128))

# Dynamic shapes using None for unknown dimensions
dynamic_reshape = is_op("relax.reshape")(wildcard()).has_shape((None, 128))
```

**Signature:**
```python
pattern.has_shape(shape: tuple) -> DFPattern
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple` | Expected shape. Use `None` for unknown/dynamic dimensions. |

### 23.3.3 Attribute Constraint (has_attr)

Restricts the match to operator calls with specific attribute values.

```python
# Match conv2d with specific stride
conv2d_stride1 = is_op("relax.nn.conv2d")(wildcard(), wildcard()).has_attr({
    "strides": [1, 1],
})

# Match conv2d with kernel size and stride
conv2d_3x3 = is_op("relax.nn.conv2d")(wildcard(), wildcard()).has_attr({
    "kernel_size": [3, 3],
    "strides": [1, 1],
    "padding": [1, 1],
})

# Match matmul with transpose_a
matmul_trans = is_op("relax.matmul")(wildcard(), wildcard()).has_attr({
    "transpose_a": True,
})
```

**Signature:**
```python
pattern.has_attr(attrs: dict) -> DFPattern
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `attrs` | `dict` | Dictionary of attribute name-value pairs to match |

### 23.3.4 StructInfo Constraint (has_struct_info)

Restricts the match based on the full struct info of the expression. This is the most flexible constraint type.

```python
from tvm.relax import StructInfo

# Match tensor with specific shape and dtype
pat = wildcard().has_struct_info(
    relax.TensorStructInfo((128, 128), "float32")
)

# Match any tensor
tensor_pat = wildcard().has_struct_info(
    relax.TensorStructInfo(None, "float32")
)

# Match a tuple struct info
tuple_pat = wildcard().has_struct_info(
    relax.TupleStructInfo([relax.TensorStructInfo((128,), "float32")])
)
```

**Signature:**
```python
pattern.has_struct_info(sinfo: StructInfo) -> DFPattern
```

### 23.3.5 Custom Constraint (has_check)

Applies an arbitrary Python function as a constraint. The function receives the matched expression and returns `True` or `False`.

```python
# Custom check: only match if the output rank is at least 2
def rank_check(expr):
    if hasattr(expr.struct_info, "ndim"):
        return expr.struct_info.ndim >= 2
    return False

pat = is_op("relax.matmul")(wildcard(), wildcard()).has_check(rank_check)

# Custom check: only match if buffer size exceeds threshold
def large_tensor_check(expr, threshold=1024):
    if hasattr(expr.struct_info, "shape"):
        shape = expr.struct_info.shape
        total = 1
        for dim in shape:
            if isinstance(dim, tvm.tir.IntImm):
                total *= dim.value
            else:
                return True  # Dynamic shape, allow match
        return total >= threshold
    return False

large_matmul = is_op("relax.matmul")(wildcard(), wildcard()).has_check(large_tensor_check)
```

**Signature:**
```python
pattern.has_check(check_fn: Callable[[Expr], bool]) -> DFPattern
```

---

## 23.4 Logical Combinators

DPL provides logical operators to combine patterns. These operators create new patterns that match when the logical condition is satisfied.

### 23.4.1 OR Pattern (|)

Matches if either the left or right pattern matches.

```python
# Match either relu or gelu activation
activation = is_op("relax.nn.relu")(wildcard()) | is_op("relax.nn.gelu")(wildcard())

# Match either add or subtract
add_or_sub = is_op("relax.add")(wildcard(), wildcard()) | is_op("relax.subtract")(wildcard(), wildcard())

# Chain multiple ORs
any_activation = (
    is_op("relax.nn.relu")(wildcard()) |
    is_op("relax.nn.gelu")(wildcard()) |
    is_op("relax.nn.silu")(wildcard()) |
    is_op("relax.nn.tanh")(wildcard())
)
```

**Usage:**
```python
combined = pattern1 | pattern2
```

### 23.4.2 AND Pattern (&)

Matches only if both patterns match the same expression.

```python
# Match add that is also a constant expression
add_const = is_op("relax.add")(wildcard(), wildcard()) & is_const()

# Match float16 matmul with specific shape
fp16_square_matmul = (
    is_op("relax.matmul")(wildcard(), wildcard()).has_dtype("float16") &
    is_op("relax.matmul")(wildcard(), wildcard()).has_shape((128, 128))
)

# Practical example: match add where one operand is a constant (bias)
x = wildcard()
bias = is_const()
add_with_bias = is_op("relax.add")(x, bias)
```

**Usage:**
```python
combined = pattern1 & pattern2
```

### 23.4.3 NOT Pattern (~)

Matches expressions that do NOT match the inner pattern.

```python
# Match anything that is NOT a relu
not_relu = ~is_op("relax.nn.relu")(wildcard())

# Match add that is NOT followed by relu
add_no_relu = is_op("relax.add")(wildcard(), wildcard())
# (Combined with sequence patterns for context)
```

**Usage:**
```python
negated = ~pattern
```

### 23.4.4 Combining Logical Operators

Logical operators can be combined to create complex matching conditions:

```python
# Match relu or gelu, but only float16
activation_fp16 = (
    (is_op("relax.nn.relu")(wildcard()) | is_op("relax.nn.gelu")(wildcard()))
    & wildcard().has_dtype("float16")
)

# Match add or subtract, but not if one operand is constant
binary_arith = (
    (is_op("relax.add")(wildcard(), wildcard()) | is_op("relax.subtract")(wildcard(), wildcard()))
    & ~is_const()
)
```

---

## 23.5 Sequence Patterns

Sequence patterns describe the dataflow relationship between expressions. They specify how the output of one expression flows into the input of another.

### 23.5.1 used_by (^)

The `used_by` operator (`^`) specifies that the left pattern's result is used by the right pattern. The left pattern's result may also be used by other expressions.

```python
a = wildcard()
b = wildcard()

# a is used by b (a may also be used elsewhere)
seq = a ^ b

# Example: matmul is used by relu
x = wildcard()
w = wildcard()
matmul = is_op("relax.matmul")(x, w)
relu = is_op("relax.nn.relu")(matmul)

# Equivalent to: relu is a call with matmul as an argument
# But ^ also matches if matmul is stored in a variable first:
#   lv = R.matmul(x, w)
#   gv = R.relu(lv)
# The ^ operator handles variable indirection
```

### 23.5.2 only_used_by (>>)

The `only_used_by` operator (`>>`) specifies that the left pattern's result is used ONLY by the right pattern. No other expression may consume the left pattern's output.

```python
a = wildcard()
b = wildcard()

# a is ONLY used by b
seq = a >> b

# Example: matmul is exclusively consumed by add (bias add)
x = wildcard()
w = wildcard()
b = wildcard()
matmul = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(matmul, b)

# This only matches if matmul's result is not used anywhere else
exclusive = matmul >> add
```

### 23.5.3 Chaining Sequence Patterns

Sequence patterns can be chained to describe multi-step dataflow:

```python
# Full linear layer pattern: matmul -> add -> relu
x = wildcard()
w = wildcard()
b = wildcard()

matmul = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(matmul, b)
relu = is_op("relax.nn.relu")(add)

# Chain with exclusive usage
# matmul only used by add, add only used by relu
full_pattern = matmul >> add >> relu

# Or with non-exclusive usage (intermediate results may be shared)
full_pattern_relaxed = matmul ^ add ^ relu
```

### 23.5.4 Sequence Pattern with DataflowBlock

Sequence patterns handle variable indirection within DataflowBlocks automatically. A DataflowBlock in Relax introduces local DataflowVars that bind intermediate results:

```python
# Relax IR:
# @R.function
# def func(x: R.Tensor((128, 128), "float32"), w: R.Tensor((128, 128), "float32")):
#     with R.dataflow():
#         lv1 = R.matmul(x, w)        # matmul bound to DataflowVar lv1
#         lv2 = R.add(lv1, bias)       # add uses lv1
#         lv3 = R.nn.relu(lv2)         # relu uses lv2
#         gv = R.output(lv3)
#     return gv

# The sequence pattern matches through these variable bindings:
x = wildcard()
w = wildcard()
b = wildcard()
matmul = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(matmul, b)
relu = is_op("relax.nn.relu")(add)
chain = matmul >> add >> relu
# This pattern matches the above IR
```

---

## 23.6 High-level Pattern Helpers

DPL provides pre-built pattern helpers for common sub-graph structures that appear frequently in deep learning models.

### 23.6.1 make_fused_bias_activation_pattern

Creates a pattern for a common sequence: convolution (or linear) followed by optional bias addition and optional activation.

```python
from tvm.relax.dpl import make_fused_bias_activation_pattern

# Conv2d + bias + relu
pattern = make_fused_bias_activation_pattern(
    "relax.nn.conv2d",
    with_bias=True,
    activation="relax.nn.relu",
)

# Conv2d + bias (no activation)
pattern = make_fused_bias_activation_pattern(
    "relax.nn.conv2d",
    with_bias=True,
    activation=None,
)

# Conv2d + gelu (no bias)
pattern = make_fused_bias_activation_pattern(
    "relax.nn.conv2d",
    with_bias=False,
    activation="relax.nn.gelu",
)

# Linear (matmul) + bias + relu
pattern = make_fused_bias_activation_pattern(
    "relax.matmul",
    with_bias=True,
    activation="relax.nn.relu",
)
```

**Signature:**
```python
def make_fused_bias_activation_pattern(
    op_name: str,
    with_bias: bool = False,
    activation: str = None,
) -> DFPattern
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `op_name` | `str` | The main operator (e.g., `"relax.nn.conv2d"`, `"relax.matmul"`) |
| `with_bias` | `bool` | Whether to include a bias addition after the main op |
| `activation` | `str` or `None` | Activation function name (e.g., `"relax.nn.relu"`) |

### 23.6.2 make_conv2d_pattern

A specialized helper for conv2d patterns with common configurations:

```python
# Basic conv2d
conv2d = make_conv2d_pattern()

# Conv2d with padding constraint
conv2d_pad1 = make_conv2d_pattern(
    attrs={"padding": [1, 1, 1, 1]}
)
```

### 23.6.3 make_residual_block_pattern

Creates patterns for residual (skip-connection) blocks:

```python
# residual_add = main_branch + skip_connection
# Useful for ResNet-style blocks
from tvm.relax.dpl import *

inp = wildcard()
main_branch = is_op("relax.nn.conv2d")(inp, wildcard())
skip = inp  # skip connection
residual_add = is_op("relax.add")(main_branch, skip)
```

### 23.6.4 Common Model Patterns

#### Attention Block

```python
# Multi-head attention: Q*K^T / sqrt(d) -> softmax -> *V
q, k, v = wildcard(), wildcard(), wildcard()
scale = wildcard()

matmul_qk = is_op("relax.matmul")(q, k)
divide = is_op("relax.divide")(matmul_qk, scale)
softmax = is_op("relax.nn.softmax")(divide)
matmul_sv = is_op("relax.matmul")(softmax, v)

attention = matmul_qk >> divide >> softmax >> matmul_sv
```

#### Layer Normalization

```python
# LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
x = wildcard()
mean = is_op("relax.mean")(x)
sub = is_op("relax.subtract")(x, mean)
var = is_op("relax.variance")(x)
add_eps = is_op("relax.add")(var, wildcard())
sqrt = is_op("relax.sqrt")(add_eps)
divide = is_op("relax.divide")(sub, sqrt)
mul_gamma = is_op("relax.multiply")(divide, wildcard())
add_beta = is_op("relax.add")(mul_gamma, wildcard())

layernorm = sub >> divide >> mul_gamma >> add_beta
```

#### MLP Block

```python
# MLP: x -> linear1 -> gelu -> linear2
x = wildcard()
w1, b1 = wildcard(), wildcard()
w2, b2 = wildcard(), wildcard()

linear1 = is_op("relax.linear")(x, w1, b1)
gelu = is_op("relax.nn.gelu")(linear1)
linear2 = is_op("relax.linear")(gelu, w2, b2)

mlp = linear1 >> gelu >> linear2
```

---

## 23.7 Matching APIs

Once a pattern is constructed, it can be matched against Relax expressions.

### 23.7.1 Boolean Match (pattern.match)

Returns `True` if the pattern matches the expression, `False` otherwise.

```python
from tvm.relax.dpl import *

x = wildcard()
y = wildcard()
add_pat = is_op("relax.add")(x, y)

# Match against an expression
if add_pat.match(expr):
    print("Pattern matched!")
```

**Signature:**
```python
def match(self, expr: relax.Expr) -> bool
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `relax.Expr` | The expression to match against |

**Return value:** `True` if the pattern matches, `False` otherwise.

### 23.7.2 Extract Matched Expressions (extract_matched_expr)

Returns a dictionary mapping pattern objects to the matched expressions, or `None` if no match.

```python
from tvm.relax.dpl import *

x = wildcard()
w = wildcard()
b = wildcard()
matmul = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(matmul, b)

# Extract matched expressions
result = add.extract_matched_expr(expr)
if result is not None:
    print(f"Input: {result[x]}")
    print(f"Weight: {result[w]}")
    print(f"Bias: {result[b]}")
    print(f"Matmul result: {result[matmul]}")
    print(f"Full matched expression: {result[add]}")
```

**Signature:**
```python
def extract_matched_expr(self, expr: relax.Expr, var2val: dict = None) -> dict or None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `relax.Expr` | The expression to match against |
| `var2val` | `dict` | Optional mapping from Var to its bound value |

**Return value:** A dictionary `{DFPattern: relax.Expr}` mapping each sub-pattern to the expression it matched, or `None` if the pattern does not match.

### 23.7.3 Using var2val for Cross-Binding Matching

When matching within a DataflowBlock, intermediate results are bound to DataflowVars. The `var2val` mapping tells the matcher to "look through" these bindings:

```python
from tvm.relax.analysis import get_var2val
from tvm.relax.dpl import *

# Get the var2val mapping for the function
var2val = get_var2val(func)

# Now matching can see through variable bindings
result = pattern.extract_matched_expr(expr, var2val=var2val)
```

The `get_var2val` function returns a dictionary mapping each `Var` or `DataflowVar` in the function to the expression it is bound to. This is essential for matching patterns that span multiple bindings in a DataflowBlock.

### 23.7.4 Matching Across an Entire Function

```python
def find_all_matches(pattern, func):
    """Find all expressions in a function that match the pattern."""
    from tvm.relax.analysis import get_var2val

    var2val = get_var2val(func)
    matches = []

    def visit(expr):
        result = pattern.extract_matched_expr(expr, var2val=var2val)
        if result is not None:
            matches.append(result)

    # Visit all expressions in the function
    from tvm.relax.analysis import post_order_visit
    post_order_visit(func, visit)

    return matches
```

---

## 23.8 Rewriting APIs

DPL provides three levels of rewriting APIs, from simple to complex.

### 23.8.1 rewrite_call

The simplest rewriting API. It matches a pattern against call expressions and replaces them. The rewriter function receives the matched expression and a dictionary of matchings, and returns the replacement expression.

**Signature:**
```python
def rewrite_call(
    pattern: DFPattern,
    rewriter: Callable[[Expr, dict], Expr],
    func: relax.Function,
) -> relax.Function
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `DFPattern` | The pattern to match |
| `rewriter` | `Callable` | Function `(matched_expr, matchings) -> new_expr` |
| `func` | `relax.Function` | The function to rewrite |

**Return value:** A new `relax.Function` with matched patterns replaced.

**Basic Example:**

```python
from tvm.relax.dpl import rewrite_call

# Pattern: reshape(x, shape)
inp = wildcard()
shape_pat = wildcard()
reshape_pat = is_op("relax.reshape")(inp, shape_pat)

def rewriter(matched_expr, matchings):
    # Replace reshape with a custom implementation
    return relax.op.call_pure_packed(
        "my_reshape",
        matchings[inp],
        matchings[shape_pat],
        sinfo_args=matched_expr.struct_info,
    )

new_func = rewrite_call(reshape_pat, rewriter, func)
```

**Practical Example -- Fuse Matmul + Bias:**

```python
from tvm.relax.dpl import *
from tvm import relax

# Pattern: add(matmul(x, w), bias)
x = wildcard()
w = wildcard()
b = wildcard()
matmul = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(matmul, b)

def fuse_matmul_bias(matched_expr, matchings):
    # Replace with fused operation
    return relax.op.call_pure_packed(
        "fused_matmul_bias",
        matchings[x],
        matchings[w],
        matchings[b],
        sinfo_args=relax.TensorStructInfo(matched_expr.struct_info),
    )

new_func = rewrite_call(add, fuse_matmul_bias, func)
```

**Practical Example -- Replace Activation:**

```python
# Pattern: relu(x)
inp = wildcard()
relu = is_op("relax.nn.relu")(inp)

def replace_with_gelu(matched_expr, matchings):
    return relax.op.nn.gelu(matchings[inp])

new_func = rewrite_call(relu, replace_with_gelu, func)
```

### 23.8.2 rewrite_bindings with PatternContext

For rewrites that involve multiple bindings across a DataflowBlock, `rewrite_bindings` provides more control. It uses a `PatternContext` to define patterns and their relationships, and the rewriter can replace multiple bindings at once.

**Signature:**
```python
def rewrite_bindings(
    ctx: PatternContext,
    rewriter: Callable[[dict, dict], dict],
    func: relax.Function,
) -> relax.Function
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `PatternContext` | Context containing pattern definitions |
| `rewriter` | `Callable` | Function `(matchings, bindings) -> dict[Var, Expr]` |
| `func` | `relax.Function` | The function to rewrite |

**Basic Example:**

```python
from tvm.relax.dpl import rewrite_bindings, PatternContext, wildcard, is_op

with PatternContext() as ctx:
    inp_pat = wildcard()
    w1_pat = wildcard()
    w2_pat = wildcard()

    matmul1 = is_op("relax.matmul")(inp_pat, w1_pat)
    matmul2 = is_op("relax.matmul")(inp_pat, w2_pat)

    def rewriter(matchings, bindings):
        # Replace both matmul operations with a fused operation
        new_expr = relax.op.call_pure_packed(
            "fused_dual_matmul",
            matchings[inp_pat],
            matchings[w1_pat],
            matchings[w2_pat],
            sinfo_args=[
                relax.TensorStructInfo(matchings[matmul1].struct_info),
                relax.TensorStructInfo(matchings[matmul2].struct_info),
            ],
        )
        # Return mapping from old vars to new expressions
        return {
            bindings[matmul1]: relax.TupleGetItem(new_expr, 0),
            bindings[matmul2]: relax.TupleGetItem(new_expr, 1),
        }

    new_func = rewrite_bindings(ctx, rewriter, func)
```

**PatternContext Details:**

The `PatternContext` collects all pattern definitions created within its scope. When matching, it ensures that shared sub-patterns (like `inp_pat` in the example above) bind to the same expression across all patterns.

```python
# PatternContext ensures consistency
with PatternContext() as ctx:
    # Both patterns share the same input pattern
    shared_input = wildcard()
    path_a = is_op("relax.nn.conv2d")(shared_input, wildcard())
    path_b = is_op("relax.nn.max_pool2d")(shared_input)

    # When matching, shared_input must bind to the same expression
    # in both path_a and path_b
```

### 23.8.3 @R.rewriter Decorator

The most declarative rewriting API. It uses Relax's `R.function` syntax to define both the pattern and the replacement. The rewriter is defined as a class with two methods: `pattern` describes what to match, and `replacement` describes what to generate.

**Signature:**
```python
@R.rewriter
class MyRewrite:
    @R.function
    def pattern(args...):
        # Describe the pattern using Relax operators
        ...

    @R.function
    def replacement(args...):
        # Describe the replacement using Relax operators
        ...
```

**Example: Replace add with custom implementation:**

```python
from tvm import relax
from tvm.relax import R

@R.rewriter
class FastAddRewrite:
    @R.function
    def pattern(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
        C = R.add(A, B)
        return C

    @R.function
    def replacement(A: R.Tensor([16], "float32"), B: R.Tensor([16], "float32")):
        C = R.call_pure_packed(
            "my_fast_add", A, B,
            sinfo_args=R.Tensor([16], "float32")
        )
        return C

# Apply the rewriter to a module
rewritten_mod = FastAddRewrite(mod)
```

**Example: Fuse matmul + bias + relu:**

```python
@R.rewriter
class FuseLinearReLU:
    @R.function
    def pattern(
        x: R.Tensor(("m", "k"), "float32"),
        w: R.Tensor(("n", "k"), "float32"),
        b: R.Tensor(("n",), "float32"),
    ):
        matmul = R.matmul(x, w)
        bias = R.add(matmul, b)
        relu = R.nn.relu(bias)
        return relu

    @R.function
    def replacement(
        x: R.Tensor(("m", "k"), "float32"),
        w: R.Tensor(("n", "k"), "float32"),
        b: R.Tensor(("n",), "float32"),
    ):
        result = R.call_pure_packed(
            "fused_linear_relu",
            x, w, b,
            sinfo_args=R.Tensor(("m", "n"), "float32"),
        )
        return result

# Apply
rewritten_mod = FuseLinearReLU(mod)
```

**Composing Rewriters:**

Multiple rewriters can be composed using the pipe operator:

```python
@R.rewriter
class RewriteA:
    # ... pattern and replacement

@R.rewriter
class RewriteB:
    # ... pattern and replacement

# Apply both rewrites in sequence
combined = RewriteA | RewriteB
rewritten_mod = combined(mod)
```

The pipe operator creates a pipeline where `RewriteA` is applied first, then `RewriteB` is applied to the result.

**Multiple pattern-replacement pairs:**

A single rewriter class can contain multiple pattern-replacement pairs:

```python
@R.rewriter
class MyOptimizations:
    @R.function
    def pattern1(A: R.Tensor(("n",), "float32")):
        B = R.nn.relu(A)
        return B

    @R.function
    def replacement1(A: R.Tensor(("n",), "float32")):
        B = R.call_pure_packed("fast_relu", A, sinfo_args=R.Tensor(("n",), "float32"))
        return B

    @R.function
    def pattern2(A: R.Tensor(("n",), "float32")):
        B = R.nn.gelu(A)
        return B

    @R.function
    def replacement2(A: R.Tensor(("n",), "float32")):
        B = R.call_pure_packed("fast_gelu", A, sinfo_args=R.Tensor(("n",), "float32"))
        return B

# Both patterns are applied
rewritten_mod = MyOptimizations(mod)
```

---

## 23.9 Pass Integration

DPL patterns integrate directly with Relax transformation passes. This section describes the integration points.

### 23.9.1 FusionPattern

A `FusionPattern` bundles a DFPattern with metadata for use by the `FuseOpsByPattern` pass. It adds a name, optional annotations for identifying sub-patterns, and an optional check function.

**Signature:**
```python
class FusionPattern:
    def __init__(
        self,
        name: str,
        pattern: DFPattern,
        annotation_patterns: dict = None,
        check: Callable = None,
        attrs_getter: Callable = None,
    ):
        ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique name for this fusion pattern |
| `pattern` | `DFPattern` | The DPL pattern to match |
| `annotation_patterns` | `dict[str, DFPattern]` | Named sub-patterns to annotate |
| `check` | `Callable[[PatternCheckContext], bool]` | Additional check function |
| `attrs_getter` | `Callable[[dict], dict]` | Extract attributes for the fused group |

**Example:**

```python
from tvm.relax.transform import FusionPattern, PatternCheckContext

# Build the pattern
inp = wildcard()
w = wildcard()
b = wildcard()
matmul = is_op("relax.matmul")(inp, w)
add = is_op("relax.add")(matmul, b)

# Create FusionPattern with annotations and check
def check_fn(ctx: PatternCheckContext) -> bool:
    # Only fuse if the matmul dimensions are large enough
    matmul_expr = ctx.annotated_expr["matmul"]
    if hasattr(matmul_expr.struct_info, "shape"):
        return True  # Accept all for this example
    return False

pattern = FusionPattern(
    name="cutlass.matmul_bias",
    pattern=add,
    annotation_patterns={"matmul": matmul, "bias": b},
    check=check_fn,
)
```

### 23.9.2 PatternCheckContext

When a `FusionPattern` includes a check function, the function receives a `PatternCheckContext` object with detailed information about the match.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `matched_expr` | `relax.Expr` | The root expression that matched the pattern |
| `annotated_expr` | `dict[str, relax.Expr]` | Map from annotation names to matched expressions |
| `matched_bindings` | `dict[Var, relax.Expr]` | Map from variables to their bound expressions |
| `var_usages` | `dict[Var, Sequence[Var]]` | Map from variables to their usage sites |
| `value_to_bound_var` | `dict[relax.Expr, Var]` | Map from expressions to their bound variables |

**Example check functions:**

```python
def check_matmul_size(ctx: PatternCheckContext) -> bool:
    """Only fuse matmul+bias for large matrices."""
    matmul_expr = ctx.annotated_expr.get("matmul")
    if matmul_expr is None:
        return False

    # Check that both M and N dimensions are at least 16
    shape = matmul_expr.struct_info.shape
    if len(shape) != 2:
        return False

    m, n = shape
    if isinstance(m, tvm.tir.IntImm) and isinstance(n, tvm.tir.IntImm):
        return m.value >= 16 and n.value >= 16
    return True  # Dynamic shapes, allow fusion


def check_single_consumer(ctx: PatternCheckContext) -> bool:
    """Ensure intermediate results have a single consumer."""
    for var, usages in ctx.var_usages.items():
        if len(usages) > 1:
            return False
    return True


def check_no_cross_block_usage(ctx: PatternCheckContext) -> bool:
    """Ensure all matched expressions are in the same DataflowBlock."""
    # This is a more complex check that may require inspecting
    # the IR structure
    return True
```

### 23.9.3 FuseOpsByPattern

The `FuseOpsByPattern` pass applies a list of fusion patterns to a Relax module, creating fused function groups.

**Signature:**
```python
class FuseOpsByPattern:
    def __init__(
        self,
        patterns: list,
        bind_constants: bool = False,
        annotate_codegen: bool = False,
    ):
        ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `patterns` | `list[FusionPattern]` | List of fusion patterns to apply |
| `bind_constants` | `bool` | Whether to bind matched constants as arguments |
| `annotate_codegen` | `bool` | Whether to annotate fused groups for codegen dispatch |

**Example -- Backend Dispatch with FusionPatterns:**

```python
from tvm.relax.transform import FuseOpsByPattern, FusionPattern

# Define patterns for a custom backend
patterns = [
    FusionPattern(
        name="cutlass.matmul_bias_relu",
        pattern=make_fused_bias_activation_pattern(
            "relax.matmul",
            with_bias=True,
            activation="relax.nn.relu",
        ),
    ),
    FusionPattern(
        name="cutlass.matmul_bias_gelu",
        pattern=make_fused_bias_activation_pattern(
            "relax.matmul",
            with_bias=True,
            activation="relax.nn.gelu",
        ),
    ),
    FusionPattern(
        name="cutlass.conv2d_bias_relu",
        pattern=make_fused_bias_activation_pattern(
            "relax.nn.conv2d",
            with_bias=True,
            activation="relax.nn.relu",
        ),
    ),
]

# Apply fusion
mod = FuseOpsByPattern(
    patterns=patterns,
    bind_constants=True,
    annotate_codegen=True,
)(mod)
```

**Pipeline Integration:**

The typical placement of `FuseOpsByPattern` in the Relax compilation pipeline:

```python
# Full compilation pipeline with pattern-based fusion
pipeline = tvm.relax.transform.Sequential([
    # Frontend import
    tvm.relax.transform.FuseOps(),              # Graph-level fusion
    tvm.relax.transform.FuseOpsByPattern(       # Pattern-based fusion
        patterns=my_patterns,
        annotate_codegen=True,
    ),
    tvm.relax.transform.LegalizeOps(),           # Lower to TIR
    tvm.relax.transform.FuseTIR(),               # TIR-level fusion
    tvm.relax.transform.DeadCodeElimination(),   # Clean up
])

optimized_mod = pipeline(mod)
```

### 23.9.4 Pattern Priority and Ordering

When multiple patterns could match the same sub-graph, the first matching pattern in the list wins. This means the ordering of patterns in the list matters:

```python
# More specific patterns should come first
patterns = [
    # Specific: matmul + bias + relu
    FusionPattern(name="matmul_bias_relu", pattern=specific_pattern),
    # General: matmul + bias (any activation)
    FusionPattern(name="matmul_bias", pattern=general_pattern),
    # Most general: matmul alone
    FusionPattern(name="matmul", pattern=matmul_pattern),
]
```

### 23.9.5 Bind Constants

When `bind_constants=True`, matched constant expressions are bound as function arguments in the fused function rather than being inlined:

```python
# With bind_constants=False:
# Fused function gets the actual constant values inlined
# def fused_matmul_bias(x, w, const_0_5):
#     ...

# With bind_constants=True:
# Constants are passed as arguments
# def fused_matmul_bias(x, w, bias):
#     ...
```

This is useful for backends that handle constants specially (e.g., embedding them in kernel binaries).

---

## 23.10 Advanced Pattern Techniques

### 23.10.1 Recursive Patterns

For matching recursive or self-referential sub-graph structures:

```python
# Match a chain of repeated operations (e.g., multiple conv-bn-relu blocks)
# DPL does not directly support recursive patterns, but you can
# match a fixed-depth chain

# 3-layer conv-relu chain
def make_n_layer_conv_relu(n):
    inp = wildcard()
    current = inp
    for _ in range(n):
        w = wildcard()
        conv = is_op("relax.nn.conv2d")(current, w)
        current = is_op("relax.nn.relu")(conv)
    return current, inp

pattern, input_pat = make_n_layer_conv_relu(3)
```

### 23.10.2 Conditional Rewriting with Pattern Contexts

```python
from tvm.relax.dpl import rewrite_bindings, PatternContext

with PatternContext() as ctx:
    # Define patterns that share a common input
    shared_input = wildcard()

    # Branch 1: conv2d path
    conv_w = wildcard()
    conv = is_op("relax.nn.conv2d")(shared_input, conv_w)

    # Branch 2: identity path
    identity = shared_input

    # Merge: add branches
    merge = is_op("relax.add")(conv, identity)

    def rewriter(matchings, bindings):
        # Replace with a fused residual block
        fused = relax.op.call_pure_packed(
            "fused_residual_conv2d",
            matchings[shared_input],
            matchings[conv_w],
            sinfo_args=relax.TensorStructInfo(matchings[merge].struct_info),
        )
        return {bindings[merge]: fused}

    new_func = rewrite_bindings(ctx, rewriter, func)
```

### 23.10.3 Shape-Aware Rewriting

```python
from tvm import relax, tir
from tvm.relax.dpl import *

def make_shape_aware_rewrite():
    """Rewrite add into broadcast_add when shapes differ."""
    a = wildcard()
    b = wildcard()

    # Match add operation
    add_pat = is_op("relax.add")(a, b)

    def rewriter(matched_expr, matchings):
        a_expr = matchings[a]
        b_expr = matchings[b]

        # Get shapes
        a_shape = a_expr.struct_info.shape
        b_shape = b_expr.struct_info.shape

        # Check if broadcasting is needed
        if a_shape != b_shape:
            return relax.op.call_pure_packed(
                "optimized_broadcast_add",
                a_expr, b_expr,
                sinfo_args=matched_expr.struct_info,
            )
        else:
            return relax.op.call_pure_packed(
                "optimized_add",
                a_expr, b_expr,
                sinfo_args=matched_expr.struct_info,
            )

    return add_pat, rewriter

pattern, rewriter = make_shape_aware_rewrite()
new_func = rewrite_call(pattern, rewriter, func)
```

### 23.10.4 Multi-Output Patterns

```python
# Pattern that matches a sub-graph producing multiple outputs
x = wildcard()
w_conv = wildcard()
w_bn = wildcard()
b_bn = wildcard()

conv = is_op("relax.nn.conv2d")(x, w_conv)
bn_scale = is_op("relax.multiply")(conv, w_bn)
bn_shift = is_op("relax.add")(bn_scale, b_bn)

# Match both outputs: scaled and shifted
with PatternContext() as ctx:
    # The context tracks all patterns defined within it
    pass
```

---

## 23.11 Complete Examples

### 23.11.1 Cutlass Backend Integration

This example shows a complete pattern-based backend integration for NVIDIA Cutlass:

```python
import tvm
from tvm import relax
from tvm.relax.dpl import *
from tvm.relax.transform import FusionPattern, FuseOpsByPattern, PatternCheckContext

# Step 1: Define patterns for Cutlass kernel dispatch
def make_cutlass_patterns():
    patterns = []

    # Matmul + bias + relu
    inp = wildcard()
    w = wildcard()
    b = wildcard()
    matmul = is_op("relax.matmul")(inp, w)
    bias_add = is_op("relax.add")(matmul, b)
    relu = is_op("relax.nn.relu")(bias_add)

    def check_cutlass_eligible(ctx: PatternCheckContext) -> bool:
        """Check if the matched pattern can use Cutlass."""
        matmul_expr = ctx.annotated_expr.get("matmul")
        if matmul_expr is None:
            return False
        # Cutlass requires float16 or int8
        dtype = matmul_expr.struct_info.dtype
        return dtype in ["float16", "int8"]

    patterns.append(FusionPattern(
        name="cutlass.matmul_bias_relu",
        pattern=relu,
        annotation_patterns={"matmul": matmul, "bias": b, "activation": relu},
        check=check_cutlass_eligible,
    ))

    # Matmul + bias (no activation)
    inp = wildcard()
    w = wildcard()
    b = wildcard()
    matmul = is_op("relax.matmul")(inp, w)
    bias_add = is_op("relax.add")(matmul, b)

    patterns.append(FusionPattern(
        name="cutlass.matmul_bias",
        pattern=bias_add,
        annotation_patterns={"matmul": matmul, "bias": b},
        check=check_cutlass_eligible,
    ))

    # Matmul alone
    inp = wildcard()
    w = wildcard()
    matmul = is_op("relax.matmul")(inp, w)

    patterns.append(FusionPattern(
        name="cutlass.matmul",
        pattern=matmul,
        annotation_patterns={"matmul": matmul},
    ))

    return patterns

# Step 2: Apply in the pipeline
patterns = make_cutlass_patterns()
mod = FuseOpsByPattern(
    patterns=patterns,
    bind_constants=True,
    annotate_codegen=True,
)(mod)
```

### 23.11.2 Operator Decomposition

Using DPL to decompose complex operators into simpler ones:

```python
from tvm.relax.dpl import rewrite_call, wildcard, is_op
from tvm import relax

# Pattern: layer norm
x = wildcard()
gamma = wildcard()
beta = wildcard()
eps = wildcard()

mean = is_op("relax.mean")(x)
sub = is_op("relax.subtract")(x, mean)
var = is_op("relax.variance")(x)
add_eps = is_op("relax.add")(var, eps)
sqrt = is_op("relax.sqrt")(add_eps)
div = is_op("relax.divide")(sub, sqrt)
mul = is_op("relax.multiply")(div, gamma)
add_beta = is_op("relax.add")(mul, beta)

# Replace with a single layer_norm op
def replace_with_layer_norm(matched_expr, matchings):
    return relax.op.nn.layer_norm(
        matchings[x],
        matchings[gamma],
        matchings[beta],
        eps=matchings[eps],
    )

new_func = rewrite_call(add_beta, replace_with_layer_norm, func)
```

### 23.11.3 Quantization Pattern

Using DPL to detect and rewrite quantization patterns:

```python
from tvm.relax.dpl import *

# Pattern: quantize (scale * x, then cast to int8)
x = wildcard()
scale = is_const()
scaled = is_op("relax.multiply")(x, scale)
rounded = is_op("relax.round")(scaled)
clipped = is_op("relax.clip")(rounded)
quantized = is_op("relax.cast")(clipped).has_dtype("int8")

def replace_with_quantize(matched_expr, matchings):
    return relax.op.call_pure_packed(
        "quantize_int8",
        matchings[x],
        matchings[scale],
        sinfo_args=relax.TensorStructInfo(
            matched_expr.struct_info.shape,
            "int8",
        ),
    )

new_func = rewrite_call(quantized, replace_with_quantize, func)
```

### 23.11.4 Attention Kernel Fusion

```python
from tvm.relax.dpl import *

# Multi-head attention fusion pattern
q = wildcard()
k = wildcard()
v = wildcard()
scale = wildcard()

# Q * K^T
transpose_k = is_op("relax.permute_dims")(k)
qk_matmul = is_op("relax.matmul")(q, transpose_k)

# Scale
scaled = is_op("relax.multiply")(qk_matmul, scale)

# Softmax
softmax = is_op("relax.nn.softmax")(scaled)

# Attention output: softmax * V
attn_output = is_op("relax.matmul")(softmax, v)

# Optionally match output projection
out_w = wildcard()
out_proj = is_op("relax.matmul")(attn_output, out_w)

# Check function for head dimension constraint
def check_attention(ctx: PatternCheckContext) -> bool:
    """Ensure head dimension is compatible with flash attention."""
    softmax_expr = ctx.annotated_expr.get("softmax")
    if softmax_expr is None:
        return False
    return True  # Simplified check

# Create fusion pattern
from tvm.relax.transform import FusionPattern

flash_attn_pattern = FusionPattern(
    name="flash_attention",
    pattern=attn_output,
    annotation_patterns={
        "q": q, "k": k, "v": v, "scale": scale,
        "softmax": softmax, "output": attn_output,
    },
    check=check_attention,
)
```

---

## 23.12 DPL Pattern Types Reference

### 23.12.1 Pattern Class Hierarchy

```
DFPattern (base class)
  +-- ExprPattern        -- Wraps a Relax expression
  +-- VarPattern         -- Matches Var nodes
  +-- DataflowVarPattern -- Matches DataflowVar nodes
  +-- GlobalVarPattern   -- Matches GlobalVar nodes
  +-- ConstantPattern    -- Matches Constant nodes
  +-- CallPattern        -- Matches function calls
  +-- FunctionPattern    -- Matches Function nodes
  +-- TuplePattern       -- Matches Tuple nodes
  +-- TupleGetItemPattern -- Matches TupleGetItem nodes
  +-- IfPattern          -- Matches If nodes
  +-- OrPattern          -- Logical OR of two patterns
  +-- AndPattern         -- Logical AND of two patterns
  +-- NotPattern         -- Logical NOT of a pattern
  +-- ShapePattern       -- Matches shape expressions
  +-- TypePattern        -- Matches type constraints
```

### 23.12.2 Pattern Construction Functions Reference

| Function | Returns | Description |
|----------|---------|-------------|
| `wildcard()` | `DFPattern` | Match any expression |
| `is_op(name)` | `DFPattern` | Match operator call |
| `is_const()` | `DFPattern` | Match constant |
| `is_var(name)` | `DFPattern` | Match Var node |
| `is_dfv(name)` | `DFPattern` | Match DataflowVar node |
| `is_gv(name)` | `DFPattern` | Match GlobalVar node |
| `is_tuple(fields)` | `DFPattern` | Match tuple |
| `is_tuple_get_item(tup, idx)` | `DFPattern` | Match tuple element |
| `is_call_tir(func, args)` | `DFPattern` | Match call_tir |
| `is_call_dps_packed(func, args)` | `DFPattern` | Match call_dps_packed |
| `is_call_packed(func, args)` | `DFPattern` | Match call_packed |
| `is_graph(expr)` | `DFPattern` | Match entire graph pattern |

### 23.12.3 Constraint Methods Reference

| Method | Parameter | Description |
|--------|-----------|-------------|
| `.has_dtype(dtype)` | `str` | Constrain to specific data type |
| `.has_shape(shape)` | `tuple` | Constrain to specific shape |
| `.has_attr(attrs)` | `dict` | Constrain operator attributes |
| `.has_struct_info(si)` | `StructInfo` | Constrain to struct info |
| `.has_check(fn)` | `Callable` | Custom constraint function |

### 23.12.4 Combinator Operators Reference

| Operator | Syntax | Description |
|----------|--------|-------------|
| OR | `p1 \| p2` | Match if either pattern matches |
| AND | `p1 & p2` | Match if both patterns match |
| NOT | `~p` | Match if pattern does not match |
| used_by | `p1 ^ p2` | p1's output is used by p2 |
| only_used_by | `p1 >> p2` | p1's output is only used by p2 |

---

## 23.13 Rewriting API Reference

### 23.13.1 rewrite_call

```python
def rewrite_call(
    pattern: DFPattern,
    rewriter: Callable[[relax.Expr, dict[DFPattern, relax.Expr]], relax.Expr],
    func: relax.Function,
) -> relax.Function
```

- **When to use**: Simple, local rewrites where the pattern and replacement are single expressions.
- **Scope**: Matches within individual call expressions.
- **Performance**: Fastest rewriting API; minimal overhead.

### 23.13.2 rewrite_bindings

```python
def rewrite_bindings(
    ctx: PatternContext,
    rewriter: Callable[
        [dict[DFPattern, relax.Expr], dict[DFPattern, relax.Var]],
        dict[relax.Var, relax.Expr]
    ],
    func: relax.Function,
) -> relax.Function
```

- **When to use**: Rewrites that span multiple variable bindings within a DataflowBlock.
- **Scope**: Matches across bindings in a DataflowBlock.
- **Performance**: More overhead than `rewrite_call` due to cross-binding analysis.

### 23.13.3 @R.rewriter

```python
@R.rewriter
class MyRewriter:
    @R.function
    def pattern(...) -> ...:
        ...

    @R.function
    def replacement(...) -> ...:
        ...
```

- **When to use**: Declarative, readable rewrites. Best for documentation and complex patterns.
- **Scope**: Matches within and across DataflowBlocks.
- **Performance**: Similar to `rewrite_bindings` but with additional overhead for pattern extraction from the R.function definition.

### 23.13.4 Comparison of Rewriting APIs

| Feature | rewrite_call | rewrite_bindings | @R.rewriter |
|---------|-------------|-----------------|-------------|
| Single-expression rewrite | Yes | Yes | Yes |
| Multi-binding rewrite | No | Yes | Yes |
| Declarative syntax | No | No | Yes |
| Composability (pipe) | Manual | Manual | Built-in (`\|`) |
| Pattern extraction | Explicit | Explicit | Automatic |
| Debugging | Easy | Medium | Harder |
| Learning curve | Low | Medium | High |

---

## 23.14 Debugging DPL Patterns

### 23.14.1 Testing Pattern Matching

Always test patterns in isolation before integrating with passes:

```python
import tvm
from tvm import relax
from tvm.relax.dpl import *
from tvm.script import relax as R

# Define a test function
@R.function
def test_func(
    x: R.Tensor((128, 64), "float32"),
    w: R.Tensor((32, 64), "float32"),
    b: R.Tensor((32,), "float32"),
):
    with R.dataflow():
        lv1 = R.matmul(x, w)
        lv2 = R.add(lv1, b)
        gv = R.output(lv2)
    return gv

# Build pattern
x_pat = wildcard()
w_pat = wildcard()
b_pat = wildcard()
matmul_pat = is_op("relax.matmul")(x_pat, w_pat)
add_pat = is_op("relax.add")(matmul_pat, b_pat)

# Test matching
from tvm.relax.analysis import get_var2val
var2val = get_var2val(test_func)

# Try matching against each expression
for name, expr in var2val.items():
    result = add_pat.extract_matched_expr(expr, var2val=var2val)
    if result is not None:
        print(f"Matched at {name}:")
        for pat, matched in result.items():
            print(f"  {pat} -> {matched}")
```

### 23.14.2 Common Pitfalls

1. **Missing var2val**: If pattern matching fails unexpectedly, ensure you are passing `var2val=get_var2val(func)` to `extract_matched_expr`.

2. **Wrong operator name**: Operator names must exactly match the registered name. Use `relax.op.get(op_name)` to verify a name is valid.

3. **Pattern ordering**: In sequence patterns, the order matters. `a >> b` means "a is only used by b", not "b is only used by a".

4. **Shared wildcard semantics**: The same wildcard object used in multiple positions must match the same expression. Use different wildcard instances for different capture positions.

```python
# CORRECT: Different wildcards for different positions
x = wildcard()  # captures input
w = wildcard()  # captures weight
b = wildcard()  # captures bias
matmul = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(matmul, b)

# WRONG: Same wildcard used for different positions
# This requires both arguments to be the same expression!
shared = wildcard()
matmul = is_op("relax.matmul")(shared, shared)  # Both args must be equal!
```

5. **Tuple vs positional arguments**: When matching operators with attribute arguments (like `reshape`), distinguish between positional arguments and attributes:

```python
# reshape has shape as a positional argument
inp = wildcard()
shape = wildcard()
reshape = is_op("relax.reshape")(inp, shape)

# conv2d has strides as an attribute, not a positional argument
conv = is_op("relax.nn.conv2d")(wildcard(), wildcard()).has_attr({"strides": [1, 1]})
```

### 23.14.3 Pattern Visualization

For complex patterns, it can help to visualize the pattern structure:

```python
def print_pattern_structure(pattern, indent=0):
    """Print a human-readable representation of a DPL pattern."""
    prefix = "  " * indent
    pat_type = type(pattern).__name__
    print(f"{prefix}{pat_type}")

    if hasattr(pattern, 'args'):
        for arg in pattern.args:
            print_pattern_structure(arg, indent + 1)

    if hasattr(pattern, 'pattern'):
        print_pattern_structure(pattern.pattern, indent + 1)
```

---

## 23.15 Performance Considerations

### 23.15.1 Pattern Complexity

Pattern matching time is proportional to the pattern complexity and the number of expressions in the function. For large models:

1. **Use specific patterns**: Patterns with constraints (type, shape, attributes) are faster because they can reject non-matching expressions early.

2. **Avoid deep nesting**: Very deep pattern chains (10+ levels) can be slow. Consider breaking them into multiple shallower patterns.

3. **Order patterns by specificity**: When using `FuseOpsByPattern`, put the most specific patterns first to reduce backtracking.

### 23.15.2 Rewriting Overhead

1. **Minimize pattern re-creation**: Build patterns once and reuse them across functions in the module.

2. **Batch rewrites**: Use a single `rewrite_call` or `rewrite_bindings` call with a complex pattern rather than multiple calls with simple patterns.

3. **Use rewrite_call when possible**: `rewrite_call` is faster than `rewrite_bindings` because it does not need to analyze cross-binding relationships.

### 23.15.3 FusionPattern Check Functions

Check functions in `FusionPattern` are called for every potential match. Keep them fast:

```python
# Good: fast check
def fast_check(ctx: PatternCheckContext) -> bool:
    dtype = ctx.annotated_expr.get("matmul").struct_info.dtype
    return dtype == "float16"

# Bad: slow check (analyzing the entire module)
def slow_check(ctx: PatternCheckContext) -> bool:
    # This would be very expensive
    for func in ctx.module.functions.values():
        analyze_function(func)
    return True
```

---

## 23.16 Design Rationale

### 23.16.1 Why Pattern-Based Rewriting?

TVM chose pattern-based rewriting for Relax because:

1. **Separation of concerns**: Patterns describe what to match; rewriters describe what to generate. This separation makes both easier to understand and test.

2. **Composability**: Patterns can be combined, chained, and reused without modification. This enables building libraries of patterns for different backends.

3. **Declarative nature**: Pattern definitions are close to the mathematical description of the transformations they implement.

4. **Backend integration**: External backend vendors can define fusion patterns for their hardware without modifying TVM's core code.

### 23.16.2 DPL vs. Relay's Pattern Language

DPL is inspired by Relay's pattern language but includes several improvements:

1. **Sequence operators**: `used_by` and `only_used_by` explicitly model dataflow, which was implicit in Relay patterns.

2. **StructInfo constraints**: DPL can match on Relax's richer type system (StructInfo), not just Relay's simpler type system.

3. **PatternContext**: The context-based matching in `rewrite_bindings` enables multi-pattern consistency that was difficult in Relay.

4. **@R.rewriter**: The declarative rewriter syntax provides a cleaner API than Relay's callback-based rewriting.

### 23.16.3 Limitations

1. **No recursive patterns**: DPL cannot express patterns of unbounded depth (e.g., "a chain of N relu operations"). Fixed-depth patterns must be used instead.

2. **No negation in sequence**: You cannot express "match A followed by NOT B". The NOT combinator only works on individual expressions.

3. **No cross-function patterns**: DPL patterns match within a single Relax function. Cross-function optimization requires module-level passes.

4. **Attribute matching requires exact values**: `has_attr` matches exact attribute values. There is no built-in support for attribute ranges or predicates.

---

## 23.17 Summary

The Dataflow Pattern Language is the primary mechanism for graph-level transformations in TVM's Relax frontend. Its key concepts are:

| Concept | Description |
|---------|-------------|
| Pattern | Describes a sub-graph structure to search for |
| Constraint | Additional condition (type, shape, attribute) on a pattern |
| Combinator | Logical or sequence operator combining patterns |
| Matching | Finding expressions that satisfy a pattern |
| Rewriting | Replacing matched patterns with new structures |
| FusionPattern | Pattern bundled with metadata for FuseOpsByPattern |

The three rewriting APIs provide increasing levels of control:

| API | Use Case |
|-----|----------|
| `rewrite_call` | Simple, single-expression rewrites |
| `rewrite_bindings` | Multi-binding rewrites within DataflowBlocks |
| `@R.rewriter` | Declarative rewrites with composable pipelines |
