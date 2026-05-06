# Pytrees in JAX

This document provides an exhaustive reference for JAX's pytree system. Pytrees (python trees) are the core data structure abstraction in JAX, used to handle arbitrarily nested collections of arrays in a uniform way. Understanding pytrees is essential for working with JAX transformations like `vmap`, `grad`, `jit`, and `pmap`.

---

## Table of Contents

1. [What is a Pytree?](#1-what-is-a-pytree)
2. [Default Pytree Types](#2-default-pytree-types)
3. [jax.tree.map](#3-jaxtreemap)
4. [jax.tree.leaves and jax.tree.structure](#4-jaxtreeleaves-and-jaxtreestructure)
5. [jax.tree.flatten and jax.tree.unflatten](#5-jaxtreeflatten-and-jaxtreeunflatten)
6. [jax.tree.map_with_path](#6-jaxtreemap_with_path)
7. [jax.tree.transpose](#7-jaxtreetranspose)
8. [jax.tree.reduce](#8-jaxtreereduce)
9. [Key Paths](#9-key-paths)
10. [Custom Pytree Node Registration](#10-custom-pytree-node-registration)
11. [Pytrees with JAX Transformations](#11-pytrees-with-jax-transformations)
12. [Common Patterns and Gotchas](#12-common-patterns-and-gotchas)
13. [Advanced Pytree Utilities](#13-advanced-pytree-utilities)

---

## 1. What is a Pytree?

A pytree is a tree-like structure built from container-like Python objects. It is a generalization of nested containers (dicts, lists, tuples, etc.) where:

- **Leaf nodes** are "regular" values, typically JAX arrays (or scalars, NumPy arrays, etc.).
- **Internal nodes** are container objects that hold other pytrees.

JAX treats pytrees as a fundamental abstraction. Every JAX transformation operates on pytrees:

- `jax.grad` returns a pytree of gradients matching the structure of the differentiated argument.
- `jax.jit` traces through pytree-structured arguments.
- `jax.vmap` can vectorize over pytree-structured inputs.

```python
import jax
import jax.numpy as jnp

# A pytree: nested dict of arrays
params = {
    "encoder": {
        "weights": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "bias": jnp.array([0.1, 0.2]),
    },
    "decoder": {
        "weights": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        "bias": jnp.array([0.3, 0.4]),
    },
}

# JAX can inspect the tree structure
print(f"Number of leaves: {jax.tree.leaves(params).__len__()}")
print(f"Tree structure: {jax.tree.structure(params)}")
```

### Simple Example

```python
import jax
import jax.numpy as jnp

# Various pytrees
tree1 = [1, 2, 3]                          # list of ints
tree2 = (jnp.array(1.0), jnp.array(2.0))   # tuple of arrays
tree3 = {"a": jnp.zeros(3), "b": jnp.ones(2)}  # dict of arrays
tree4 = [jnp.array([1, 2]), {"x": jnp.array(3)}]  # mixed nesting

# All of these are valid pytrees that JAX can process
for i, tree in enumerate([tree1, tree2, tree3, tree4], 1):
    leaves = jax.tree.leaves(tree)
    print(f"tree{i}: {len(leaves)} leaves, types: {[type(l).__name__ for l in leaves]}")
```

---

## 2. Default Pytree Types

JAX classifies Python types into two categories:

### Node Types (Internal Nodes)

These are treated as internal nodes of the pytree. Their children are their elements, and their structure is preserved.

| Type | Children |
|------|----------|
| `dict` | Values (keys are part of the structure) |
| `list` | Elements |
| `tuple` | Elements |
| `namedtuple` | Fields |
| `OrderedDict` | Values |
| `defaultdict` | Values |

### Leaf Types

These are treated as leaves (not recursively traversed):

| Type | Notes |
|------|-------|
| `jax.Array` | JAX arrays are always leaves |
| `numpy.ndarray` | NumPy arrays are leaves |
| `int`, `float`, `bool`, `complex` | Python scalars |
| `str`, `bytes` | Strings are leaves |
| `None` | None is a leaf |
| Most other types | Unless registered as a pytree node |

```python
import jax
import jax.numpy as jnp
from collections import namedtuple, OrderedDict

# Demonstrate different pytree types
Point = namedtuple("Point", ["x", "y"])

trees = {
    "dict": {"a": 1, "b": 2},
    "list": [1, 2, 3],
    "tuple": (1, 2, 3),
    "namedtuple": Point(x=1, y=2),
    "OrderedDict": OrderedDict([("a", 1), ("b", 2)]),
    "nested": {"layer1": [1, 2], "layer2": (3, {"deep": 4})},
    "array_leaf": {"data": jnp.array([1, 2, 3])},
}

for name, tree in trees.items():
    structure = jax.tree.structure(tree)
    leaves = jax.tree.leaves(tree)
    print(f"{name}: {structure}, leaves={leaves}")
```

### What Is NOT a Pytree Node (Treated as Leaf)

```python
import jax
import jax.numpy as jnp
import numpy as np

# These are all treated as single leaves
print(jax.tree.leaves(jnp.array([1, 2, 3])))     # [Array([1, 2, 3])]
print(jax.tree.leaves(np.array([1, 2, 3])))       # [array([1, 2, 3])]
print(jax.tree.leaves("hello"))                    # ['hello']
print(jax.tree.leaves(None))                       # [None]
print(jax.tree.leaves(42))                         # [42]

# Sets are NOT pytree nodes -- treated as leaves
s = {1, 2, 3}
print(jax.tree.leaves(s))  # [{1, 2, 3}]  -- the whole set is one leaf!
```

---

## 3. jax.tree.map

`jax.tree.map` applies a function to every leaf of a pytree, preserving the tree structure. This is the most commonly used pytree operation.

### Signature

```python
jax.tree.map(f, tree, *rest, is_leaf=None)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp

# Double every leaf
tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
doubled = jax.tree.map(lambda x: x * 2, tree)
print(doubled)
# {'a': Array([2., 4.]), 'b': Array([6., 8.])}
```

### Multiple Trees (Element-Wise)

```python
import jax
import jax.numpy as jnp

tree1 = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.1])}
tree2 = {"w": jnp.array([3.0, 4.0]), "b": jnp.array([0.2])}

# Element-wise addition
summed = jax.tree.map(lambda a, b: a + b, tree1, tree2)
print(f"Sum: {summed}")
# {'w': Array([4., 6.]), 'b': Array([0.3])}

# Element-wise multiply
product = jax.tree.map(lambda a, b: a * b, tree1, tree2)
print(f"Product: {product}")
```

### SGD Update Pattern

```python
import jax
import jax.numpy as jnp

def sgd_update(params, grads, lr=0.01):
    """Standard SGD update using tree_map."""
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)

params = {
    "layer1": {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.1, 0.2])},
    "layer2": {"w": jnp.array([[5.0, 6.0]]), "b": jnp.array([0.3])},
}
grads = jax.tree.map(lambda x: x * 0.1, params)  # fake gradients

new_params = sgd_update(params, grads)
print(f"Updated layer1 w:\n{new_params['layer1']['w']}")
```

### Initialize Zeros

```python
import jax
import jax.numpy as jnp

def zeros_like_tree(tree):
    """Create a tree of zeros with the same structure."""
    return jax.tree.map(jnp.zeros_like, tree)

params = {
    "w1": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    "b1": jnp.array([0.1, 0.2]),
    "w2": jnp.array([5.0, 6.0]),
    "b2": jnp.array([0.3]),
}

zeros = zeros_like_tree(params)
print(f"Zeros w1 shape: {zeros['w1'].shape}")
print(f"Zeros b1: {zeros['b1']}")
```

### tree.map with is_leaf

```python
import jax
import jax.numpy as jnp

tree = {
    "a": jnp.array([1.0, 2.0]),
    "metadata": {"lr": 0.01, "step": 100},  # nested dict without arrays
}

# Default: recursively traverses into the nested dict
leaves_default = jax.tree.leaves(tree)
print(f"Default leaves: {leaves_default}")

# With is_leaf: stop at dicts that don't contain arrays
def is_leaf(x):
    return isinstance(x, dict) and all(not isinstance(v, jax.Array) for v in x.values())

leaves_custom = jax.tree.leaves(tree, is_leaf=is_leaf)
print(f"Custom leaves: {leaves_custom}")
```

### Multiple Trees Must Have Compatible Structure

```python
import jax
import jax.numpy as jnp

# Trees must have the same structure when using multiple trees
tree1 = {"a": 1, "b": 2}
tree2 = {"a": 10, "b": 20}

# Works: same structure
result = jax.tree.map(lambda x, y: x + y, tree1, tree2)
print(f"Result: {result}")  # {'a': 11, 'b': 22}

# ERROR: different structure
# tree3 = {"a": 1, "c": 2}
# result = jax.tree.map(lambda x, y: x + y, tree1, tree3)  # ValueError!
```

---

## 4. jax.tree.leaves and jax.tree.structure

### jax.tree.leaves

`jax.tree.leaves` extracts all leaf values from a pytree as a flat list.

```python
import jax
import jax.numpy as jnp

tree = {
    "layer1": {
        "weights": jnp.array([[1.0, 2.0]]),
        "bias": jnp.array([0.1]),
    },
    "layer2": {
        "weights": jnp.array([[3.0, 4.0]]),
        "bias": jnp.array([0.2]),
    },
}

leaves = jax.tree.leaves(tree)
print(f"Number of leaves: {len(leaves)}")
for i, leaf in enumerate(leaves):
    print(f"  Leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}")
```

### jax.tree.structure

`jax.tree.structure` returns a `PyTreeDef` object that encodes the tree structure without the leaf values. This is useful for comparing tree structures.

```python
import jax
import jax.numpy as jnp

tree1 = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
tree2 = {"a": jnp.array(10.0), "b": jnp.array(20.0)}
tree3 = {"a": jnp.array(1.0), "c": jnp.array(2.0)}

struct1 = jax.tree.structure(tree1)
struct2 = jax.tree.structure(tree2)
struct3 = jax.tree.structure(tree3)

print(f"struct1 == struct2: {struct1 == struct2}")  # True (same structure)
print(f"struct1 == struct3: {struct1 == struct3}")  # False (different keys)
print(f"struct1: {struct1}")
```

### Counting Leaves and Checking Structure

```python
import jax
import jax.numpy as jnp

params = {
    "encoder": [
        (jnp.zeros((784, 256)), jnp.zeros(256)),
        (jnp.zeros((256, 128)), jnp.zeros(128)),
    ],
    "decoder": [
        (jnp.zeros((128, 256)), jnp.zeros(256)),
        (jnp.zeros((256, 784)), jnp.zeros(784)),
    ],
}

# Count total parameters
leaves = jax.tree.leaves(params)
total_params = sum(leaf.size for leaf in leaves)
print(f"Total parameters: {total_params}")

# Check structure matches (for gradient compatibility)
grad_template = jax.tree.map(jnp.zeros_like, params)
print(f"Same structure: {jax.tree.structure(params) == jax.tree.structure(grad_template)}")
```

---

## 5. jax.tree.flatten and jax.tree.unflatten

`flatten` converts a pytree into a flat list of leaves plus a `PyTreeDef` structure descriptor. `unflatten` reconstructs a pytree from a list of leaves and a structure.

### Signature

```python
leaves, treedef = jax.tree.flatten(tree)
tree = jax.tree.unflatten(treedef, leaves)
```

### Basic Round-Trip

```python
import jax
import jax.numpy as jnp

tree = {
    "weights": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    "bias": jnp.array([0.1, 0.2]),
    "config": {"lr": 0.01, "momentum": 0.9},  # Note: floats are leaves too
}

# Flatten
leaves, treedef = jax.tree.flatten(tree)
print(f"Leaves ({len(leaves)}):")
for leaf in leaves:
    print(f"  {type(leaf).__name__}: {leaf}")

print(f"\nTreedef: {treedef}")

# Unflatten
reconstructed = jax.tree.unflatten(treedef, leaves)
print(f"\nReconstructed matches original: {jax.tree.equal(tree, reconstructed)}")
```

### Modifying Leaves and Reconstructing

```python
import jax
import jax.numpy as jnp

params = {
    "w1": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    "b1": jnp.array([0.1, 0.2]),
    "w2": jnp.array([[5.0, 6.0]]),
    "b2": jnp.array([0.3]),
}

# Flatten, modify, and unflatten
leaves, treedef = jax.tree.flatten(params)

# Apply L2 regularization: scale each array by 0.99
modified_leaves = [leaf * 0.99 for leaf in leaves]

# Reconstruct with same structure
regularized = jax.tree.unflatten(treedef, modified_leaves)
print(f"Original w1:\n{params['w1']}")
print(f"Regularized w1:\n{regularized['w1']}")
```

### Concatenating Leaves from Two Trees

```python
import jax
import jax.numpy as jnp

def concat_trees(tree1, tree2):
    """Concatenate corresponding leaves from two identically-structured trees."""
    leaves1, treedef = jax.tree.flatten(tree1)
    leaves2, _ = jax.tree.flatten(tree2)
    concat_leaves = [jnp.concatenate([l1, l2]) for l1, l2 in zip(leaves1, leaves2)]
    return jax.tree.unflatten(treedef, concat_leaves)

tree1 = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
tree2 = {"a": jnp.array([4.0, 5.0]), "b": jnp.array([6.0])}

result = concat_trees(tree1, tree2)
print(f"Concatenated a: {result['a']}")  # [1, 2, 4, 5]
print(f"Concatenated b: {result['b']}")  # [3, 6]
```

### Saving and Loading Parameters

```python
import jax
import jax.numpy as jnp
import numpy as np

def save_params(params, filepath):
    """Save parameters to a numpy file."""
    leaves, treedef = jax.tree.flatten(params)
    # Convert to numpy for serialization
    np_leaves = [np.asarray(leaf) for leaf in leaves]
    np.savez(filepath, *np_leaves, treedef=str(treedef))

def load_params(filepath, treedef_str):
    """Load parameters from a numpy file."""
    data = np.load(filepath)
    leaves = [data[f'arr_{i}'] for i in range(len(data.files) - 1)]
    # Reconstruct (treedef_str would need to be parsed in practice)
    # This is a simplified example
    return leaves

# Example usage
params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}
leaves, treedef = jax.tree.flatten(params)
print(f"Treedef for later reconstruction: {treedef}")
```

### flatten with is_leaf

```python
import jax
import jax.numpy as jnp

tree = {"data": jnp.array([1.0]), "meta": {"nested": {"deep": jnp.array(2.0)}}}

# Default: flattens all the way down
leaves_default, struct_default = jax.tree.flatten(tree)
print(f"Default: {len(leaves_default)} leaves")

# Stop at dicts: treat certain dicts as leaves
leaves_custom, struct_custom = jax.tree.flatten(
    tree,
    is_leaf=lambda x: isinstance(x, dict) and "nested" in str(x)
)
print(f"Custom: {len(leaves_custom)} leaves")
```

---

## 6. jax.tree.map_with_path

`jax.tree.map_with_path` is like `jax.tree.map`, but the function also receives the path (key path) to each leaf. This is useful when you need to know where each leaf came from in the tree structure.

### Signature

```python
jax.tree.map_with_path(f, tree, *rest, is_leaf=None)
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax.tree import DictKey, SequenceKey

params = {
    "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.1])},
    "layer2": {"w": jnp.array([3.0]), "b": jnp.array([0.2])},
}

def print_path_and_value(path, value):
    """Print the path and value of each leaf."""
    path_str = "/".join(str(k) for k in path)
    print(f"  {path_str}: shape={value.shape}, value={value}")
    return value

print("Parameter structure:")
jax.tree.map_with_path(print_path_and_value, params)
# layer1/w: shape=(2,), value=[1. 2.]
# layer1/b: shape=(1,), value=[0.1]
# layer2/w: shape=(1,), value=[3.]
# layer2/b: shape=(1,), value=[0.2]
```

### Differentiated Treatment Based on Path

```python
import jax
import jax.numpy as jnp

params = {
    "weights": jnp.array([1.0, 2.0, 3.0]),
    "bias": jnp.array([0.1]),
    "batch_norm_scale": jnp.array([1.0]),
    "batch_norm_offset": jnp.array([0.0]),
}

def selective_regularize(path, value):
    """Apply L2 regularization only to weights, not biases or BN params."""
    # Check if the path indicates this is a weight parameter
    key_names = [str(k) for k in path]
    is_weight = any("weight" in k for k in key_names)
    is_bn = any("batch_norm" in k for k in key_names)

    if is_weight:
        return value + 0.01 * value  # regularize
    else:
        return value  # leave unchanged

regularized = jax.tree.map_with_path(selective_regularize, params)
print(f"Original weights: {params['weights']}")
print(f"Regularized weights: {regularized['weights']}")
print(f"Unchanged bias: {regularized['bias']}")
```

### Debugging Pytree Structure

```python
import jax
import jax.numpy as jnp

model_state = {
    "params": {
        "linear1": {"w": jnp.zeros((4, 8)), "b": jnp.zeros(8)},
        "linear2": {"w": jnp.zeros((8, 2)), "b": jnp.zeros(2)},
    },
    "optimizer_state": {
        "mu": {
            "linear1": {"w": jnp.zeros((4, 8)), "b": jnp.zeros(8)},
            "linear2": {"w": jnp.zeros((8, 2)), "b": jnp.zeros(2)},
        },
        "nu": {
            "linear1": {"w": jnp.zeros((4, 8)), "b": jnp.zeros(8)},
            "linear2": {"w": jnp.zeros((8, 2)), "b": jnp.zeros(2)},
        },
    },
}

def describe(path, value):
    path_str = "/".join(str(k) for k in path)
    return f"{path_str}: shape={value.shape}"

descriptions = jax.tree.map_with_path(describe, model_state)
for desc in jax.tree.leaves(descriptions):
    print(desc)
```

### map_with_path with Multiple Trees

```python
import jax
import jax.numpy as jnp

grads = {"w": jnp.array([0.1, 0.2]), "b": jnp.array([0.01])}
params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.1])}

def update_with_logging(path, grad, param):
    """Update parameters and log the path."""
    path_str = "/".join(str(k) for k in path)
    updated = param - 0.01 * grad
    print(f"  Updating {path_str}: {param} -> {updated}")
    return updated

print("Applying gradients:")
new_params = jax.tree.map_with_path(update_with_logging, grads, params)
```

---

## 7. jax.tree.transpose

`jax.tree.transpose` restructures a pytree by swapping inner and outer grouping. It converts a tree of groups into a group of trees (or vice versa).

### Signature

```python
jax.tree.transpose(outer_treedef, inner_treedef, pytree_to_transpose)
```

### Basic Example

```python
import jax
import jax.numpy as jnp

# Structure: {layer_name: {param_type: array}}
# i.e. outer keys = layer names, inner keys = param types
layer_grouped = {
    "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.1])},
    "layer2": {"w": jnp.array([3.0, 4.0]), "b": jnp.array([0.2])},
}

# Transpose to: {param_type: {layer_name: array}}
# i.e. outer keys = param types, inner keys = layer names
outer_treedef = jax.tree.structure({"w": 0, "b": 0})  # desired inner structure
inner_treedef = jax.tree.structure({"layer1": 0, "layer2": 0})  # current inner structure

param_grouped = jax.tree.transpose(outer_treedef, inner_treedef, layer_grouped)
print(f"By parameter type:")
for param_type, layers in param_grouped.items():
    print(f"  {param_type}: {layers}")
# {'w': {'layer1': Array([1., 2.]), 'layer2': Array([3., 4.])},
#  'b': {'layer1': Array([0.1]), 'layer2': Array([0.2])}}
```

### Transpose for Multi-Optimizer State

```python
import jax
import jax.numpy as jnp

# Optimizer state grouped by parameter
# {param_name: {momentum: array, velocity: array}}
state_by_param = {
    "w1": {"m": jnp.array([0.1, 0.2]), "v": jnp.array([0.01, 0.02])},
    "w2": {"m": jnp.array([0.3]), "v": jnp.array([0.03])},
}

# Transpose to group by optimizer variable
# {momentum: {param_name: array}, velocity: {param_name: array}}
inner_treedef = jax.tree.structure({"m": 0, "v": 0})
outer_treedef = jax.tree.structure({"w1": 0, "w2": 0})

state_by_optvar = jax.tree.transpose(outer_treedef, inner_treedef, state_by_param)
print(f"Momentum state: {state_by_optvar['m']}")
print(f"Velocity state: {state_by_optvar['v']}")
```

### Transpose with Lists and Tuples

```python
import jax
import jax.numpy as jnp

# A list of training examples, each is a dict with 'input' and 'target'
batch = [
    {"input": jnp.array([1.0, 2.0]), "target": jnp.array([1.0])},
    {"input": jnp.array([3.0, 4.0]), "target": jnp.array([2.0])},
    {"input": jnp.array([5.0, 6.0]), "target": jnp.array([3.0])},
]

# Transpose to: {'input': list_of_inputs, 'target': list_of_targets}
inner_treedef = jax.tree.structure({"input": 0, "target": 0})
outer_treedef = jax.tree.structure([0, 0, 0])

grouped = jax.tree.transpose(outer_treedef, inner_treedef, batch)
print(f"Inputs: {grouped['input']}")
print(f"Targets: {grouped['target']}")
```

### Transpose Back and Forth

```python
import jax
import jax.numpy as jnp

original = {
    "layer1": {"w": jnp.array([1.0]), "b": jnp.array([0.1])},
    "layer2": {"w": jnp.array([2.0]), "b": jnp.array([0.2])},
}

inner_treedef = jax.tree.structure({"w": 0, "b": 0})
outer_treedef = jax.tree.structure({"layer1": 0, "layer2": 0})

# Transpose once
transposed = jax.tree.transpose(outer_treedef, inner_treedef, original)

# Transpose back (swap inner and outer)
restored = jax.tree.transpose(inner_treedef, outer_treedef, transposed)

print(f"Original == Restored: {jax.tree.equal(original, restored)}")
```

---

## 8. jax.tree.reduce

`jax.tree.reduce` applies a reduction function across all leaves of a pytree, combining them into a single value.

### Signature

```python
jax.tree.reduce(function, tree, initializer=None, is_leaf=None)
```

### Basic Usage: Sum All Parameters

```python
import jax
import jax.numpy as jnp

params = {
    "layer1": {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.1, 0.2])},
    "layer2": {"w": jnp.array([[5.0, 6.0]]), "b": jnp.array([0.3])},
}

# Sum of all parameter values
total_sum = jax.tree.reduce(lambda acc, x: acc + jnp.sum(x), params, 0.0)
print(f"Total parameter sum: {total_sum}")

# Count total parameters
total_count = jax.tree.reduce(lambda acc, x: acc + x.size, params, 0)
print(f"Total parameter count: {total_count}")
```

### L2 Norm of All Parameters

```python
import jax
import jax.numpy as jnp

params = {
    "w1": jnp.array([1.0, 2.0, 3.0]),
    "w2": jnp.array([4.0, 5.0]),
}

l2_norm_sq = jax.tree.reduce(lambda acc, x: acc + jnp.sum(x ** 2), params, 0.0)
l2_norm = jnp.sqrt(l2_norm_sq)
print(f"L2 norm: {l2_norm}")  # sqrt(1+4+9+16+25) = sqrt(55) ~ 7.416
```

### Maximum Value Across All Leaves

```python
import jax
import jax.numpy as jnp

params = {
    "layer1": {"w": jnp.array([[1.0, -5.0], [3.0, 0.5]]), "b": jnp.array([-0.1, 0.2])},
    "layer2": {"w": jnp.array([[7.0, -2.0]]), "b": jnp.array([0.3])},
}

global_max = jax.tree.reduce(
    lambda acc, x: jnp.maximum(acc, jnp.max(x)),
    params,
    -jnp.inf,
)
print(f"Global max value: {global_max}")  # 7.0
```

### Gradient Norm for Clipping

```python
import jax
import jax.numpy as jnp

def grad_norm(grads):
    """Compute the global gradient norm."""
    return jnp.sqrt(jax.tree.reduce(
        lambda acc, g: acc + jnp.sum(g ** 2),
        grads,
        0.0,
    ))

def clip_grads(grads, max_norm=1.0):
    """Clip gradients by global norm."""
    norm = grad_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads), norm

# Example
grads = {
    "w1": jnp.array([10.0, 20.0, 30.0]),
    "w2": jnp.array([40.0, 50.0]),
}
clipped, norm = clip_grads(grads, max_norm=1.0)
print(f"Original norm: {norm:.4f}")
print(f"Clipped norm: {grad_norm(clipped):.4f}")
```

### Custom Accumulator Pattern

```python
import jax
import jax.numpy as jnp

# Collect statistics about all arrays in a tree
def tree_stats(tree):
    """Compute min, max, mean across all leaves."""
    def reducer(acc, x):
        stats_acc = acc
        stats_new = {
            "min": jnp.min(x),
            "max": jnp.max(x),
            "mean": jnp.mean(x),
            "count": x.size,
        }
        if stats_acc is None:
            return stats_new
        # Combine: weighted mean, overall min/max
        total = stats_acc["count"] + stats_new["count"]
        return {
            "min": jnp.minimum(stats_acc["min"], stats_new["min"]),
            "max": jnp.maximum(stats_acc["max"], stats_new["max"]),
            "mean": (stats_acc["mean"] * stats_acc["count"] + stats_new["mean"] * stats_new["count"]) / total,
            "count": total,
        }
    return jax.tree.reduce(reducer, tree, None)

params = {
    "layer1": {"w": jax.random.normal(jax.random.key(0), (10, 20)), "b": jnp.zeros(20)},
    "layer2": {"w": jax.random.normal(jax.random.key(1), (20, 5)), "b": jnp.zeros(5)},
}

stats = tree_stats(params)
print(f"Global min: {stats['min']:.4f}")
print(f"Global max: {stats['max']:.4f}")
print(f"Global mean: {stats['mean']:.4f}")
print(f"Total elements: {stats['count']}")
```

---

## 9. Key Paths

When JAX flattens a pytree, it tracks the path to each leaf using key objects. These keys describe how to navigate from the root to each leaf.

### Key Types

- **`SequenceKey(i)`**: Index into a sequence (list, tuple). `i` is the integer index.
- **`DictKey(key)`**: Key into a dictionary. `key` is the dictionary key.
- **`GetAttrKey(name)`**: Attribute name (for namedtuples and custom nodes). `name` is the attribute name.
- **`FlattenedIndexKey(index)`**: Index into a flattened sequence (used for some custom types).

### Inspecting Key Paths

```python
import jax
import jax.numpy as jnp
from jax.tree import DictKey, SequenceKey, GetAttrKey

tree = {
    "weights": jnp.array([1.0, 2.0]),
    "biases": [jnp.array(0.1), jnp.array(0.2)],
}

# Get paths and leaves
paths, leaves, treedef = jax.tree.flatten_with_path(tree)
for path, leaf in zip(paths, leaves):
    key_types = [(type(k).__name__, str(k)) for k in path]
    print(f"Path: {key_types} -> {leaf}")

# Example output:
# Path: [('DictKey', 'weights')] -> [1. 2.]
# Path: [('DictKey', 'biases'), ('SequenceKey', '0')] -> 0.1
# Path: [('DictKey', 'biases'), ('SequenceKey', '1')] -> 0.2
```

### flatten_with_path

```python
import jax
import jax.numpy as jnp

tree = [
    {"name": "a", "value": jnp.array(1.0)},
    {"name": "b", "value": jnp.array(2.0)},
]

paths, leaves, treedef = jax.tree.flatten_with_path(tree)

for path, leaf in zip(paths, leaves):
    # Convert path to human-readable string
    path_str = jax.tree.keystr(path)
    print(f"  {path_str} = {leaf}")
```

### Using Key Paths for Conditional Logic

```python
import jax
import jax.numpy as jnp

params = {
    "encoder": {
        "conv1": {"w": jnp.ones((3, 3, 1, 32)), "b": jnp.zeros(32)},
        "conv2": {"w": jnp.ones((3, 3, 32, 64)), "b": jnp.zeros(64)},
    },
    "decoder": {
        "dense": {"w": jnp.ones((64, 10)), "b": jnp.zeros(10)},
    },
}

# Apply different initialization based on path
def init_by_path(path, shape_dtype):
    """Initialize parameters differently based on their path."""
    path_str = jax.tree.keystr(path)
    if "conv" in path_str and "w" in path_str:
        # He initialization for conv weights
        fan_in = shape_dtype.shape[-2]
        return jax.random.normal(jax.random.key(0), shape_dtype.shape) * jnp.sqrt(2.0 / fan_in)
    elif "dense" in path_str and "w" in path_str:
        # Xavier initialization for dense weights
        fan_in, fan_out = shape_dtype.shape
        return jax.random.normal(jax.random.key(1), shape_dtype.shape) * jnp.sqrt(2.0 / (fan_in + fan_out))
    else:
        # Zero initialization for biases
        return jnp.zeros(shape_dtype.shape)

# Get shape/dtype info
params_spec = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
initialized = jax.tree.map_with_path(init_by_path, params_spec)
print(f"conv1 w mean: {jnp.mean(initialized['encoder']['conv1']['w']):.4f}")
print(f"dense w mean: {jnp.mean(initialized['decoder']['dense']['w']):.4f}")
```

### Key Path Representation

```python
import jax
import jax.numpy as jnp
from collections import namedtuple

# Namedtuple uses GetAttrKey
Layer = namedtuple("Layer", ["weight", "bias"])

tree = {
    "layers": [
        Layer(weight=jnp.array([1.0]), bias=jnp.array([0.1])),
        Layer(weight=jnp.array([2.0]), bias=jnp.array([0.2])),
    ]
}

paths, leaves, treedef = jax.tree.flatten_with_path(tree)
for path, leaf in zip(paths, leaves):
    key_details = [(type(k).__name__, vars(k) if hasattr(k, '__dict__') else str(k)) for k in path]
    print(f"Keys: {key_details}")
    print(f"  -> Leaf: {leaf}\n")
```

---

## 10. Custom Pytree Node Registration

You can register custom Python classes as pytree nodes, allowing JAX to handle them like built-in container types.

### register_pytree_node

```python
import jax
import jax.numpy as jnp

class Linear:
    """A linear layer with weight and bias."""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __repr__(self):
        return f"Linear(weight={self.weight.shape}, bias={self.bias.shape})"

# Register as a pytree node
def flatten_linear(layer):
    """Return (children, auxiliary_data)."""
    children = (layer.weight, layer.bias)
    aux_data = None  # no auxiliary data needed
    return children, aux_data

def unflatten_linear(aux_data, children):
    """Reconstruct from children and auxiliary data."""
    weight, bias = children
    return Linear(weight, bias)

jax.tree.register_pytree_node(
    Linear,
    flatten_func=flatten_linear,
    unflatten_func=unflatten_linear,
)

# Now Linear is a valid pytree node
layer = Linear(jnp.ones((4, 3)), jnp.zeros(3))
print(f"Leaves: {jax.tree.leaves(layer)}")
print(f"Structure: {jax.tree.structure(layer)}")

# tree_map works
doubled = jax.tree.map(lambda x: x * 2, layer)
print(f"Doubled weight: {doubled.weight}")
```

### Custom Node with Auxiliary Data

```python
import jax
import jax.numpy as jnp

class NormalizedLinear:
    """Linear layer with normalization parameters."""
    def __init__(self, weight, bias, scale, offset):
        self.weight = weight
        self.bias = bias
        self.scale = scale   # differentiable
        self.offset = offset # differentiable

    def __call__(self, x):
        out = x @ self.weight + self.bias
        return out * self.scale + self.offset

def flatten_normalized(layer):
    children = (layer.weight, layer.bias, layer.scale, layer.offset)
    aux_data = None
    return children, aux_data

def unflatten_normalized(aux_data, children):
    return NormalizedLinear(*children)

jax.tree.register_pytree_node(
    NormalizedLinear,
    flatten_func=flatten_normalized,
    unflatten_func=unflatten_normalized,
)

# Use with grad
def loss_fn(layer, x, y):
    preds = layer(x)
    return jnp.mean((preds - y) ** 2)

layer = NormalizedLinear(
    weight=jnp.ones((3, 2)),
    bias=jnp.zeros(2),
    scale=jnp.ones(2),
    offset=jnp.zeros(2),
)

x = jax.random.normal(jax.random.key(0), (5, 3))
y = jax.random.normal(jax.random.key(1), (5, 2))

# Gradient is a NormalizedLinear object with gradient arrays
grad_layer = jax.grad(loss_fn)(layer, x, y)
print(f"Weight grad shape: {grad_layer.weight.shape}")
print(f"Bias grad shape: {grad_layer.bias.shape}")
print(f"Scale grad shape: {grad_layer.scale.shape}")
print(f"Offset grad shape: {grad_layer.offset.shape}")
```

### register_pytree_node_class (Decorator)

```python
import jax
import jax.numpy as jnp

@jax.tree.register_pytree_node_class
class MLP:
    """Multi-layer perceptron registered as pytree node."""

    def __init__(self, layers, activation="relu"):
        self.layers = layers          # list of (weight, bias) tuples
        self.activation = activation  # auxiliary data (not differentiated)

    def tree_flatten(self):
        """Return (children, auxiliary_data)."""
        children = self.layers
        aux_data = self.activation
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from children and auxiliary data."""
        return cls(children, aux_data)

    def __call__(self, x):
        for w, b in self.layers:
            x = x @ w + b
            if self.activation == "relu":
                x = jnp.maximum(0, x)
            elif self.activation == "tanh":
                x = jnp.tanh(x)
        return x

# Usage
model = MLP(
    layers=[
        (jax.random.normal(jax.random.key(0), (4, 8)) * 0.1, jnp.zeros(8)),
        (jax.random.normal(jax.random.key(1), (8, 2)) * 0.1, jnp.zeros(2)),
    ],
    activation="relu",
)

# JAX transformations work
x = jax.random.normal(jax.random.key(2), (3, 4))
output = model(x)
print(f"Output shape: {output.shape}")

# Gradient through the model
def loss_fn(model, x, y):
    preds = model(x)
    return jnp.mean((preds - y) ** 2)

y = jax.random.normal(jax.random.key(3), (3, 2))
grad_model = jax.grad(loss_fn)(model, x, y)
print(f"Number of layers with grads: {len(grad_model.layers)}")
```

### Custom Pytree for Optimizer State

```python
import jax
import jax.numpy as jnp

@jax.tree.register_pytree_node_class
class AdamState:
    """Adam optimizer state for a parameter tree."""
    def __init__(self, mu, nu, t):
        self.mu = mu    # first moment estimates (same structure as params)
        self.nu = nu    # second moment estimates (same structure as params)
        self.t = t      # timestep (scalar)

    def tree_flatten(self):
        children = (self.mu, self.nu)
        aux_data = self.t
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mu, nu = children
        return cls(mu, nu, aux_data)

def init_adam(params):
    """Initialize Adam state from parameter structure."""
    return AdamState(
        mu=jax.tree.map(jnp.zeros_like, params),
        nu=jax.tree.map(jnp.zeros_like, params),
        t=jnp.array(1.0),
    )

def adam_step(state, grads, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam optimization step."""
    new_mu = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state.mu, grads)
    new_nu = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g ** 2, state.nu, grads)

    mu_hat = jax.tree.map(lambda m: m / (1 - beta1 ** state.t), new_mu)
    nu_hat = jax.tree.map(lambda v: v / (1 - beta2 ** state.t), new_nu)

    updates = jax.tree.map(
        lambda m, v: lr * m / (jnp.sqrt(v) + eps), mu_hat, nu_hat
    )
    new_params = jax.tree.map(lambda p, u: p - u, params, updates)
    new_state = AdamState(new_mu, new_nu, state.t + 1)

    return new_state, new_params

# Example
params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([0.5])}
state = init_adam(params)

for _ in range(5):
    grads = jax.tree.map(lambda x: x * 0.1, params)  # dummy gradients
    state, params = adam_step(state, grads, params)

print(f"Updated params: {params}")
print(f"State timestep: {state.t}")
```

---

## 11. Pytrees with JAX Transformations

### vmap with in_axes on Pytrees

`jax.vmap` can vectorize over pytree-structured inputs. The `in_axes` argument specifies which axis to vectorize over for each leaf.

```python
import jax
import jax.numpy as jnp

def predict(params, x):
    """Single-input prediction."""
    h = x
    for w, b in params:
        h = jnp.maximum(0, h @ w + b)
    return h

params = [
    (jax.random.normal(jax.random.key(0), (4, 8)) * 0.1, jnp.zeros(8)),
    (jax.random.normal(jax.random.key(1), (8, 2)) * 0.1, jnp.zeros(2)),
]

# Vectorize over x only (params are shared)
batched_predict = jax.vmap(predict, in_axes=(None, 0))

x_batch = jax.random.normal(jax.random.key(2), (10, 4))  # 10 samples
outputs = batched_predict(params, x_batch)
print(f"Batched output shape: {outputs.shape}")  # (10, 2)
```

### vmap with Nested in_axes

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    preds = params["w"] @ x + params["b"]
    return jnp.sum((preds - y) ** 2)

params = {
    "w": jax.random.normal(jax.random.key(0), (3, 4)),
    "b": jnp.zeros(3),
}

x_batch = jax.random.normal(jax.random.key(1), (8, 4))   # batch of 8
y_batch = jax.random.normal(jax.random.key(2), (8, 3))

# vmap: params are shared, x and y are batched along axis 0
batched_loss = jax.vmap(loss_fn, in_axes=(None, 0, 0))

losses = batched_loss(params, x_batch, y_batch)
print(f"Per-sample losses shape: {losses.shape}")  # (8,)
print(f"Per-sample losses: {losses}")
```

### Per-Sample Gradients with vmap + grad

```python
import jax
import jax.numpy as jnp

def loss_fn(params, x, y):
    """Loss for a single sample."""
    pred = jnp.dot(params, x)
    return (pred - y) ** 2

# Per-sample gradients: vmap(grad(...))
per_sample_grad = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))

params = jnp.array([1.0, 2.0, 3.0])
x_batch = jnp.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
y_batch = jnp.array([1.0, 4.0, 9.0])

grads = per_sample_grad(params, x_batch, y_batch)
print(f"Per-sample gradients shape: {grads.shape}")  # (3, 3)
print(f"Gradients:\n{grads}")
```

### jit with Pytrees

```python
import jax
import jax.numpy as jnp

@jax.jit
def train_step(params, x, y, lr=0.01):
    """JIT-compiled training step with pytree params."""
    def loss_fn(params):
        preds = x @ params["w"] + params["b"]
        return jnp.mean((preds - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss

params = {
    "w": jax.random.normal(jax.random.key(0), (4, 2)) * 0.1,
    "b": jnp.zeros(2),
}
x = jax.random.normal(jax.random.key(1), (10, 4))
y = jax.random.normal(jax.random.key(2), (10, 2))

for step in range(5):
    params, loss = train_step(params, x, y)
    print(f"Step {step}: loss = {loss:.6f}")
```

### pmap with Pytrees

```python
import jax
import jax.numpy as jnp

# Note: this requires multiple devices
def replicated_train_step(params, x, y):
    """Training step for pmap (each device gets a shard of data)."""
    def loss_fn(params):
        preds = x @ params["w"] + params["b"]
        return jnp.mean((preds - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name="devices")
    new_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    return new_params, loss

# For single-device testing, just show the structure
params = {
    "w": jax.random.normal(jax.random.key(0), (4, 2)) * 0.1,
    "b": jnp.zeros(2),
}
# pmapped = jax.pmap(replicated_train_step, axis_name="devices")
# params and data would be replicated/sharded across devices
print("pmap-ready function defined")
```

---

## 12. Common Patterns and Gotchas

### Pattern: Creating Zero-Initialized Copies

```python
import jax
import jax.numpy as jnp

params = {"w1": jnp.ones((3, 4)), "b1": jnp.ones(4), "w2": jnp.ones((4, 2)), "b2": jnp.ones(2)}

# Create zero-initialized accumulator
zeros = jax.tree.map(jnp.zeros_like, params)
print(f"Zeros w1: {zeros['w1']}")

# Also works for other initializations
ones = jax.tree.map(lambda x: jnp.ones(x.shape, dtype=x.dtype), params)
random_init = jax.tree.map(lambda x: jax.random.normal(jax.random.key(0), x.shape), params)
```

### Pattern: Gradient Accumulation

```python
import jax
import jax.numpy as jnp

def accumulate_grads(grad_fn, params, data_batches):
    """Accumulate gradients over multiple mini-batches."""
    total_grads = jax.tree.map(jnp.zeros_like, params)

    for batch in data_batches:
        grads = grad_fn(params, batch)
        total_grads = jax.tree.map(lambda acc, g: acc + g, total_grads, grads)

    # Average
    n = len(data_batches)
    return jax.tree.map(lambda g: g / n, total_grads)
```

### Pattern: Weight Decay (Selective)

```python
import jax
import jax.numpy as jnp

def apply_weight_decay(params, grads, lr=0.01, weight_decay=0.001):
    """Apply weight decay only to weight matrices, not biases."""
    def maybe_decay(path, param, grad):
        # Check if this is a weight (not a bias)
        path_str = jax.tree.keystr(path)
        if "bias" not in path_str.lower():
            return grad + weight_decay * param
        return grad

    adjusted_grads = jax.tree.map_with_path(maybe_decay, params, grads)
    return jax.tree.map(lambda p, g: p - lr * g, params, adjusted_grads)
```

### Gotcha: Sets Are Not Pytree Nodes

```python
import jax
import jax.numpy as jnp

# Sets are treated as leaves, not as pytree nodes
s = {1, 2, 3}
print(jax.tree.leaves(s))  # [{1, 2, 3}] -- the whole set is one leaf!

# If you need a set-like structure, use a sorted tuple or frozenset
t = (1, 2, 3)
print(jax.tree.leaves(t))  # [1, 2, 3] -- each element is a leaf
```

### Gotcha: Strings Are Leaves

```python
import jax
import jax.numpy as jnp

# Strings are leaves, not sequences
tree = {"name": "model", "weights": jnp.array([1.0, 2.0])}
leaves = jax.tree.leaves(tree)
print(f"Leaves: {leaves}")  # ["model", Array([1., 2.])]
print(f"Number of leaves: {len(leaves)}")  # 2
```

### Gotcha: None Is a Valid Leaf

```python
import jax
import jax.numpy as jnp

# None is treated as a leaf
tree = {"a": jnp.array(1.0), "b": None}
leaves = jax.tree.leaves(tree)
print(f"Leaves: {leaves}")  # [Array(1.0), None]

# tree_map will try to apply the function to None
# doubled = jax.tree.map(lambda x: x * 2, tree)  # Error: NoneType * int

# Fix: handle None explicitly
doubled = jax.tree.map(lambda x: x * 2 if x is not None else None, tree)
print(f"Doubled: {doubled}")
```

### Gotcha: Tree Structures Must Match for Operations

```python
import jax
import jax.numpy as jnp

# This works: same structure
tree1 = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
tree2 = {"a": jnp.array(3.0), "b": jnp.array(4.0)}
result = jax.tree.map(lambda x, y: x + y, tree1, tree2)

# This FAILS: different structure (different keys)
# tree3 = {"a": jnp.array(1.0), "c": jnp.array(2.0)}
# result = jax.tree.map(lambda x, y: x + y, tree1, tree3)  # ValueError

# This FAILS: different nesting depth
# tree4 = {"a": {"inner": jnp.array(1.0)}, "b": jnp.array(2.0)}
# result = jax.tree.map(lambda x, y: x + y, tree1, tree4)  # ValueError
```

### Gotcha: Dict Key Order Matters

```python
import jax
import jax.numpy as jnp

# These two dicts have the same keys but different order
d1 = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
d2 = {"b": jnp.array(2.0), "a": jnp.array(1.0)}

# In JAX, dict key order is part of the tree structure
s1 = jax.tree.structure(d1)
s2 = jax.tree.structure(d2)
print(f"Same structure: {s1 == s2}")  # True (JAX sorts dict keys internally)

# But leaves are returned in sorted key order
print(f"d1 leaves: {jax.tree.leaves(d1)}")  # [1.0, 2.0] (a before b)
print(f"d2 leaves: {jax.tree.leaves(d2)}")  # [1.0, 2.0] (a before b, sorted)
```

### Pattern: Tree Structure Validation

```python
import jax
import jax.numpy as jnp

def assert_same_structure(tree1, tree2, name1="tree1", name2="tree2"):
    """Assert two pytrees have the same structure."""
    s1 = jax.tree.structure(tree1)
    s2 = jax.tree.structure(tree2)
    if s1 != s2:
        raise ValueError(
            f"Structure mismatch:\n"
            f"  {name1}: {s1}\n"
            f"  {name2}: {s2}"
        )

# Usage in a training loop
params = {"w": jnp.ones((3, 4)), "b": jnp.zeros(4)}
grads = jax.grad(lambda p: jnp.sum(p["w"]))(params)

assert_same_structure(params, grads, "params", "grads")  # OK
print("Structures match!")
```

### Pattern: Tree Shape Inspection

```python
import jax
import jax.numpy as jnp

def tree_shapes(tree):
    """Get shapes of all leaves in a tree."""
    return jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else type(x), tree)

def tree_dtypes(tree):
    """Get dtypes of all leaves in a tree."""
    return jax.tree.map(lambda x: x.dtype if hasattr(x, 'dtype') else type(x), tree)

params = {
    "layer1": {"w": jnp.zeros((784, 256), dtype=jnp.float32), "b": jnp.zeros(256, dtype=jnp.float32)},
    "layer2": {"w": jnp.zeros((256, 10), dtype=jnp.float16), "b": jnp.zeros(10, dtype=jnp.float16)},
}

print("Shapes:")
jax.tree.map_with_path(lambda p, s: print(f"  {jax.tree.keystr(p)}: {s}"), tree_shapes(params))

print("\nDtypes:")
jax.tree.map_with_path(lambda p, d: print(f"  {jax.tree.keystr(p)}: {d}"), tree_dtypes(params))
```

---

## 13. Advanced Pytree Utilities

### jax.tree.equal

```python
import jax
import jax.numpy as jnp

tree1 = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
tree2 = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
tree3 = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([4.0])}

print(f"tree1 == tree2: {jax.tree.equal(tree1, tree2)}")  # True
print(f"tree1 == tree3: {jax.tree.equal(tree1, tree3)}")  # False
```

### jax.tree.all

```python
import jax
import jax.numpy as jnp

# Check if all elements satisfy a condition
tree = {"a": jnp.array([True, True]), "b": jnp.array([True])}
print(f"All True: {jax.tree.all(tree)}")  # True

tree2 = {"a": jnp.array([True, False]), "b": jnp.array([True])}
print(f"All True: {jax.tree.all(tree2)}")  # False
```

### jax.tree.leaves with is_leaf

```python
import jax
import jax.numpy as jnp

# Stop recursion at certain types
tree = {
    "data": jnp.array([1.0]),
    "config": {"nested": {"deep": jnp.array(2.0)}},
}

# Default: flatten all the way
print(f"Default: {jax.tree.leaves(tree)}")

# Stop at nested dicts
print(f"Custom: {jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, dict) and 'nested' in x)}")
```

### Computing Parameter Count Efficiently

```python
import jax
import jax.numpy as jnp

def count_params(params):
    """Count total number of scalar parameters."""
    return sum(x.size for x in jax.tree.leaves(params) if hasattr(x, 'size'))

model = {
    "transformer": {
        "attention": {
            "W_q": jnp.zeros((512, 64)),
            "W_k": jnp.zeros((512, 64)),
            "W_v": jnp.zeros((512, 64)),
            "W_o": jnp.zeros((64, 512)),
        },
        "ffn": {
            "W1": jnp.zeros((512, 2048)),
            "W2": jnp.zeros((2048, 512)),
            "b1": jnp.zeros(2048),
            "b2": jnp.zeros(512),
        },
    },
}

total = count_params(model)
print(f"Total parameters: {total:,}")
```

### Deep Copy via Pytree Round-Trip

```python
import jax
import jax.numpy as jnp

def tree_copy(tree):
    """Deep copy a pytree of arrays."""
    leaves, treedef = jax.tree.flatten(tree)
    # jnp.array creates a new copy
    new_leaves = [jnp.array(leaf) for leaf in leaves]
    return jax.tree.unflatten(treedef, new_leaves)

original = {"w": jnp.array([1.0, 2.0, 3.0])}
copied = tree_copy(original)

# Modify the copy
copied["w"] = copied["w"].at[0].set(99.0)

print(f"Original: {original['w']}")  # [1, 2, 3] -- unchanged
print(f"Copied: {copied['w']}")      # [99, 2, 3]
```

### Merging Two Pytrees with Different Structures

```python
import jax
import jax.numpy as jnp

def merge_pytrees(base, update):
    """Merge update into base, only updating keys that exist in both."""
    result = {}
    for key in base:
        if key in update:
            if isinstance(base[key], dict) and isinstance(update[key], dict):
                result[key] = merge_pytrees(base[key], update[key])
            else:
                result[key] = update[key]
        else:
            result[key] = base[key]
    return result

base = {"w1": jnp.array([1.0]), "w2": jnp.array([2.0]), "w3": jnp.array([3.0])}
update = {"w1": jnp.array([10.0]), "w3": jnp.array([30.0])}

merged = merge_pytrees(base, update)
print(f"Merged: {merged}")
# {'w1': [10.0], 'w2': [2.0], 'w3': [30.0]}
```

### Summary of Pytree API

| Function | Purpose |
|----------|---------|
| `jax.tree.map(f, tree)` | Apply `f` to every leaf |
| `jax.tree.map_with_path(f, tree)` | Apply `f` with key path info |
| `jax.tree.leaves(tree)` | Get flat list of leaf values |
| `jax.tree.structure(tree)` | Get `PyTreeDef` (structure without values) |
| `jax.tree.flatten(tree)` | Get `(leaves, treedef)` |
| `jax.tree.unflatten(treedef, leaves)` | Reconstruct tree from leaves and structure |
| `jax.tree.flatten_with_path(tree)` | Get `(paths, leaves, treedef)` |
| `jax.tree.transpose(outer, inner, tree)` | Swap inner/outer tree structure |
| `jax.tree.reduce(f, tree, init)` | Reduce across all leaves |
| `jax.tree.equal(t1, t2)` | Check if two trees have equal leaves |
| `jax.tree.all(tree)` | Check if all boolean leaves are True |
| `jax.tree.keystr(path)` | Human-readable path string |
| `jax.tree.register_pytree_node` | Register custom class as pytree node |
| `jax.tree.register_pytree_node_class` | Decorator for class registration |
