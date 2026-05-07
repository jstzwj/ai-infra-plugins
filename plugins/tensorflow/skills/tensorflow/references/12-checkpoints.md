# TensorFlow Checkpoints Reference

## Table of Contents

1. [Overview](#overview)
2. [tf.train.Checkpoint](#tftraincheckpoint)
3. [tf.train.CheckpointManager](#tftraincheckpointmanager)
4. [Object-Based Saving](#object-based-saving)
5. [Variable Tracking](#variable-tracking)
6. [Checkpoint Format](#checkpoint-format)
7. [Partial Restoration](#partial-restoration)
8. [Deferred Restoration](#deferred-restoration)
9. [Training State](#training-state)
10. [Checkpoints in Distributed Training](#checkpoints-in-distributed-training)
11. [TF1 Saver Compatibility](#tf1-saver-compatibility)
12. [Checkpoint Conversion](#checkpoint-conversion)
13. [Best Practices](#best-practices)
14. [Warm Starting](#warm-starting)
15. [Internal Architecture](#internal-architecture)
16. [CheckpointOptions](#checkpointoptions)

---

## Overview

TensorFlow checkpoints provide a mechanism for saving and restoring the state of
trainable models. Unlike SavedModel (which captures the complete computation
graph plus variable state), checkpoints focus exclusively on variable values and
their relationships within the Python object graph.

The checkpoint system in TensorFlow 2 uses an **object-based** approach where
variables are identified by their position in the Python object graph rather than
by their TensorFlow name. This makes checkpoints more robust to program changes
and supports features like deferred restoration and restore-on-create.

Key characteristics:
- **Object-based**: Variables are saved with their object graph relationships,
  not their TensorFlow names.
- **Incremental**: Only variable values are saved; the graph must be recreated
  by the program.
- **Deferred loading**: Variables can be restored when they are created, allowing
  flexible loading order.
- **Robust**: Changes to variable names or graph structure don't necessarily
  break restoration.

---

## tf.train.Checkpoint

### Constructor

#### TF2 API: `tf.train.Checkpoint`

```python
tf.train.Checkpoint(root=None, **kwargs)
```

**root** (`Trackable` or `WeakRef[Trackable]`, optional):
The root object to checkpoint. When provided, all keyword arguments (including
`root` itself) are set as children of the root object rather than of the
Checkpoint.

**kwargs** (`Trackable` objects):
Keyword arguments set as attributes of the Checkpoint. Each becomes a named
dependency in the checkpoint. Values must be trackable objects (`tf.Variable`,
`tf.keras.Layer`, `tf.keras.Model`, `tf.keras.optimizers.Optimizer`, etc.) or
nested structures of trackable objects (`list`, `dict`, `tuple`).

```python
# Basic usage
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# With root object
checkpoint = tf.train.Checkpoint(root=model, optimizer=optimizer)
# In this case, optimizer is attached to root (model) not to the checkpoint

# With WeakRef
import weakref
checkpoint = tf.train.Checkpoint(root=weakref.ref(model))
```

#### TF1 API: `tf.compat.v1.train.Checkpoint`

```python
tf.compat.v1.train.Checkpoint(**kwargs)
```

In the TF1 version, there is no `root` parameter. All kwargs become direct
children of the Checkpoint object.

### Properties

**save_counter** (`tf.Variable`):
An integer variable starting at zero and incremented on each `save()` call. Used
to number checkpoints. Created lazily to support restore-on-create.

```python
checkpoint = tf.train.Checkpoint(v=tf.Variable(0.))
print(checkpoint.save_counter.numpy())  # 0
checkpoint.save('/tmp/ckpt')
print(checkpoint.save_counter.numpy())  # 1
```

### save()

```python
tf.train.Checkpoint.save(file_prefix, options=None)
```

Saves a training checkpoint with basic management:
1. Increments `save_counter`.
2. Writes checkpoint files with name `{file_prefix}-{save_counter}`.
3. Updates the `checkpoint` state file in the directory.

**file_prefix** (`str` or `PathLike`):
Prefix for checkpoint filenames. The save counter is appended.

**options** (`tf.train.CheckpointOptions`):
Optional configuration for the save operation.

**Returns**: The full path to the saved checkpoint.

```python
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
path = checkpoint.save('/tmp/training/ckpt')
# Creates: /tmp/training/ckpt-1, /tmp/training/ckpt-2, etc.
```

### write()

```python
tf.train.Checkpoint.write(file_prefix, options=None)
```

Writes a checkpoint without numbering, incrementing the save counter, or
updating metadata. Intended for use by higher-level management utilities.

**file_prefix** (`str` or `PathLike`):
The exact prefix for checkpoint files (no counter appended).

**Returns**: The full path to the checkpoint (same as `file_prefix`).

```python
checkpoint = tf.train.Checkpoint(v=tf.Variable(1.))
path = checkpoint.write('/tmp/ckpt')
# Creates: /tmp/ckpt.index, /tmp/ckpt.data-00000-of-00001
```

### restore()

```python
tf.train.Checkpoint.restore(save_path, options=None)
```

Restores checkpoint values. Supports deferred restoration: if variables don't
exist yet, their restoration is deferred until they are created.

**save_path** (`str` or `PathLike`):
Path to the checkpoint (as returned by `save()`, `write()`, or
`tf.train.latest_checkpoint()`). Can also be a SavedModel directory.

**options** (`tf.train.CheckpointOptions`):
Optional configuration for the restore operation.

**Returns**: A load status object with assertion methods.

```python
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore('/tmp/ckpt-1')
status.assert_consumed()  # Verify all objects matched

# Or with deferred creation:
checkpoint = tf.train.Checkpoint()
status = checkpoint.restore('/tmp/ckpt-1')
model = create_model()
checkpoint.model = model  # Variables restored on assignment
status.assert_consumed()
```

### read()

```python
tf.train.Checkpoint.read(save_path, options=None)
```

Reads a checkpoint written with `write()`. Unlike `restore()`, does not expect
the `save_counter` variable. Use this for checkpoints that were not created with
`save()`.

```python
checkpoint = tf.train.Checkpoint(v=tf.Variable(1.))
path = checkpoint.write('/tmp/ckpt')
# Later:
checkpoint.read(path).assert_consumed()
```

### sync()

```python
tf.train.Checkpoint.sync()
```

Waits for any outstanding asynchronous save or restore operations to complete.

---

## tf.train.CheckpointManager

### Constructor

```python
tf.train.CheckpointManager(
    checkpoint,
    directory,
    max_to_keep,
    keep_checkpoint_every_n_hours=None,
    checkpoint_name='ckpt',
    step_counter=None,
    checkpoint_interval=None,
    init_fn=None,
    last_checkpoint_step=None
)
```

**checkpoint** (`tf.train.Checkpoint`):
The Checkpoint instance to manage.

**directory** (`str`):
Directory for writing checkpoints and the state file.

**max_to_keep** (`int` or `None`):
Number of checkpoints to keep. Oldest checkpoints are deleted when exceeded.
If `None`, all checkpoints are kept (may consume significant disk space).

**keep_checkpoint_every_n_hours** (`float` or `None`):
Preserves checkpoints at this time interval even if they would otherwise be
deleted. For example, `keep_checkpoint_every_n_hours=1` keeps at most one
checkpoint per hour. Default `None` disables this preservation.

**checkpoint_name** (`str`):
Custom name for checkpoint files. Default `"ckpt"`.

**step_counter** (`tf.Variable`):
Variable for checking the current step counter. Required if `checkpoint_interval`
is not `None`.

**checkpoint_interval** (`int`):
Minimum step interval between two checkpoints. Requires `step_counter`.

**init_fn** (`callable`):
Function called to initialize the model if no checkpoints exist in `directory`.

**last_checkpoint_step** (`int`):
Starting point for `checkpoint_interval` checking. If `None`, the last
checkpoint step is set to `None`.

### Properties

**directory** (`str`): The directory path for checkpoints.

**latest_checkpoint** (`str` or `None`): Path prefix of the most recent
checkpoint. Suitable for `tf.train.Checkpoint.restore()`.

**checkpoints** (`list[str]`): Sorted list of managed checkpoint paths (oldest
to newest). Does not include checkpoints preserved by
`keep_checkpoint_every_n_hours`.

**checkpoint** (`tf.train.Checkpoint`): The managed Checkpoint object.

**checkpoint_interval** (`int` or `None`): The configured interval.

### save()

```python
CheckpointManager.save(checkpoint_number=None, check_interval=True, options=None)
```

Creates a new checkpoint and manages the lifecycle of existing ones.

**checkpoint_number** (`int`, `Variable`, or `Tensor`):
Optional number for the checkpoint. If `None` (default), uses
`checkpoint.save_counter`.

**check_interval** (`bool`):
When `True` and `checkpoint_interval` is configured, only saves if the interval
has elapsed. When `False`, always saves (unless already saved at current step).

**options** (`tf.train.CheckpointOptions`):
Options passed to the underlying save operation.

**Returns**: The path to the new checkpoint, or `None` if skipped.

### restore_or_initialize()

```python
CheckpointManager.restore_or_initialize()
```

Restores from the latest checkpoint if available, otherwise calls `init_fn`
if provided.

**Returns**: The restored checkpoint path if found, otherwise `None`.

### Example: Basic Training Loop

```python
import tensorflow as tf
import os

# Create model and optimizer
model = tf.keras.Model(...)
optimizer = tf.keras.optimizers.Adam()

# Create checkpoint and manager
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(
    checkpoint,
    directory='/tmp/training',
    max_to_keep=5,
    keep_checkpoint_every_n_hours=2)

# Restore if available
checkpoint.restore(manager.latest_checkpoint)

# Training loop
for epoch in range(num_epochs):
    for batch in dataset:
        # ... training step ...
        pass
    manager.save()
    print(f"Saved checkpoint: {manager.latest_checkpoint}")
```

### Example: Interval-Based Saving

```python
step = tf.Variable(0, name='global_step')
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
manager = tf.train.CheckpointManager(
    checkpoint,
    directory='/tmp/training',
    max_to_keep=3,
    step_counter=step,
    checkpoint_interval=1000)

for batch in dataset:
    train_step(batch)
    step.assign_add(1)
    manager.save()  # Only saves every 1000 steps
```

### Example: Custom Initialization

```python
def init_from_pretrained():
    pretrained_ckpt = tf.train.Checkpoint(model=pretrained_model)
    pretrained_ckpt.restore('/path/to/pretrained')
    checkpoint.model.set_weights(pretrained_model.get_weights())

manager = tf.train.CheckpointManager(
    checkpoint,
    directory='/tmp/fine_tuning',
    max_to_keep=3,
    init_fn=init_from_pretrained)

manager.restore_or_initialize()
```

### State Persistence

The CheckpointManager persists its state across instantiations in a
human-readable text file named `checkpoint` in the directory:

```
model_checkpoint_path: "ckpt-5"
all_model_checkpoint_paths: "ckpt-3"
all_model_checkpoint_paths: "ckpt-4"
all_model_checkpoint_paths: "ckpt-5"
all_model_checkpoint_timestamps: 1672531200.0
all_model_checkpoint_timestamps: 1672531201.0
all_model_checkpoint_timestamps: 1672531202.0
last_preserved_timestamp: 1672531000.0
```

---

## Object-Based Saving

### Trackable Base Class

All objects that participate in the checkpoint system must inherit from
`Trackable` (defined in `tensorflow.python.trackable.base`). This base class
provides:

- **Dependency tracking**: Tracks child objects via `_track_trackable()`.
- **Variable management**: Creates and manages variables via
  `_add_variable_with_custom_getter()`.
- **Serialization hooks**: `_serialize_to_tensors()` and `_restore_from_tensors()`
  for custom save/restore logic.

### NamedTrackable

Objects with named dependencies. The dependency name is the attribute name used
when assigning a trackable object:

```python
module = tf.Module()
module.v = tf.Variable(1.0)  # Named dependency "v"
module.layer = tf.keras.layers.Dense(10)  # Named dependency "layer"
```

### AutoTrackable

The `AutoTrackable` class (in `tensorflow.python.trackable.autotrackable`)
automatically tracks objects assigned as attributes:

```python
class MyModule(autotrackable.AutoTrackable):
    def __init__(self):
        self.v = tf.Variable(1.0)  # Automatically tracked
        self.sub = tf.Module()     # Automatically tracked
```

### Trackable Children

The `_trackable_children()` method returns all tracked children of an object.
This is used during save to enumerate the object graph:

```python
def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, cache=None):
    # Returns dict of {name: child_object}
    return self._trackable
```

The `save_type` parameter distinguishes between checkpoint saving and SavedModel
saving, allowing objects to expose different children for each context.

---

## Variable Tracking

### TrackableDataStructure

The `TrackableDataStructure` base class (in `tensorflow.python.trackable.data_structures`)
enables tracking of objects stored in container types.

### List

```python
from tensorflow.python.trackable import data_structures

class MyModule(tf.Module):
    def __init__(self):
        self.layers = []  # Becomes a TrackableDataStructure
        self.layers.append(tf.keras.layers.Dense(10))
        self.layers.append(tf.keras.layers.Dense(5))
```

The `List` class wraps a Python list, tracking all elements. It supports:
- `append()`, `extend()`, `insert()`
- `__getitem__()`, `__setitem__()`, `__delitem__()`
- `pop()`, `clear()`, `copy()`
- `__len__()`, `__iter__()`, `__contains__()`
- `__eq__()`, `__ne__()`, `__add__()`, `__mul__()`

### Mapping / _DictWrapper

Dictionary-like containers that track values:

```python
class MyModule(tf.Module):
    def __init__(self):
        self.layers = {
            'dense1': tf.keras.layers.Dense(10),
            'dense2': tf.keras.layers.Dense(5)
        }
```

The `_DictWrapper` class wraps a Python dict, tracking all values. It supports:
- `__getitem__()`, `__setitem__()`, `__delitem__()`
- `get()`, `keys()`, `values()`, `items()`
- `update()`, `pop()`, `setdefault()`
- `__len__()`, `__iter__()`, `__contains__()`

### NoDependency

The `NoDependency` wrapper prevents automatic tracking:

```python
from tensorflow.python.trackable.data_structures import NoDependency

class MyModule(tf.Module):
    def __init__(self):
        self.tracked_var = tf.Variable(1.0)  # Tracked
        self.untracked_var = NoDependency(tf.Variable(2.0))  # Not tracked
```

### Tuple

Tuples are also tracked when assigned as attributes. They are converted to
`List` internally for mutability during restoration.

---

## Checkpoint Format

### File Structure

A checkpoint consists of two types of files:

1. **Index file** (`{prefix}.index`):
   A string-string table mapping variable names to their locations in the data
   files. Also contains the `TrackableObjectGraph` proto under the special key
   `"_CHECKPOINTABLE_OBJECT_GRAPH"`.

2. **Data file(s)** (`{prefix}.data-NNNNN-of-MMMMM`):
   Contains serialized tensor values. For large models, data can be sharded
   across multiple files.

### TrackableObjectGraph Protocol Buffer

The object graph is serialized as:

```protobuf
message TrackableObjectGraph {
  repeated TrackableObject nodes = 1;
}

message TrackableObject {
  repeated ObjectReference children = 1;
  repeated Attribute attributes = 2;
  repeated SlotVariableReference slot_variables = 3;
}

message ObjectReference {
  string local_name = 1;
  int32 node_id = 2;
}

message Attribute {
  string full_name = 1;  // Checkpoint key
  int32 checkpoint_key = 2;
  // Optional: for registered savers
}

message SlotVariableReference {
  string slot_name = 1;
  int32 original_variable_node_id = 2;
  int32 slot_variable_node_id = 3;
}
```

### Checkpoint Keys

Each variable is identified by a checkpoint key derived from its path in the
object graph. For example:

```
model/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
model/dense/bias/.ATTRIBUTES/VARIABLE_VALUE
optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
save_counter/.ATTRIBUTES/VARIABLE_VALUE
```

The `OBJECT_GRAPH_PROTO_KEY` (`"_CHECKPOINTABLE_OBJECT_GRAPH"`) stores the
serialized object graph in the index file.

### Data File Format

Data files use the TensorFlow checkpoint tensor bundle format:
- Tensors are serialized in order of their keys.
- Each tensor entry contains: key name, shape, dtype, and raw data.
- Supports sharding: tensors can be distributed across multiple data files.
- Big-endian systems convert `tensor_content` to little-endian format.

---

## Partial Restoration

### expect_partial()

Silences warnings about incomplete checkpoint restores:

```python
status = checkpoint.restore(path)
status.expect_partial()
# No warnings about unmatched objects or values
```

This is commonly used when:
- Loading a subset of a larger checkpoint.
- Loading from a SavedModel that has extra keys.
- Loading a checkpoint saved with a different version of the model.

```python
# Loading from SavedModel
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore('/tmp/saved_model').expect_partial()
```

### assert_consumed()

Raises an exception unless all objects in the checkpoint have been matched and
all checkpointed values have corresponding Python objects:

```python
status = checkpoint.restore(path)
status.assert_consumed()
# Raises AssertionError if:
# - Any Python object has no matching checkpoint value
# - Any checkpoint value has no matching Python object
# - Any slot variables are unresolved
```

### assert_existing_objects_matched()

A weaker assertion that only checks existing Python objects have matching
checkpoint values. Does not fail for checkpoint values without Python objects:

```python
status = checkpoint.restore(path)
status.assert_existing_objects_matched()
# Fails only if existing Python objects are unmatched
# Passes even if checkpoint has extra values
```

### assert_nontrivial_match()

Asserts that something besides the root object was matched. Very weak assertion
useful for sanity checking:

```python
status = checkpoint.restore(path)
status.assert_nontrivial_match()
# Fails if nothing besides the root was matched
```

---

## Deferred Restoration

### Overview

One of the most powerful features of TF2 checkpoints is deferred restoration.
When a checkpoint is restored, variables that don't yet exist in the Python
program are queued for later restoration. When those variables are eventually
created, their values are immediately populated from the checkpoint.

### Restore-on-Create

```python
# Create an empty checkpoint
checkpoint = tf.train.Checkpoint()

# Restore before any variables exist
status = checkpoint.restore('/tmp/ckpt-1')

# Variables are restored as they are created
model = tf.keras.Model(...)
checkpoint.model = model  # All model variables immediately get restored values

optimizer = tf.keras.optimizers.Adam()
checkpoint.optimizer = optimizer  # Optimizer state immediately restored

status.assert_consumed()
```

### How Deferred Restoration Works

1. The restore operation reads the `TrackableObjectGraph` from the checkpoint.
2. For each object in the proto, it checks if a corresponding Python object
   exists in the dependency graph.
3. If the object exists, its values are restored immediately.
4. If not, the restoration is deferred. The checkpoint records the expected
   attribute name and node ID.
5. When a new trackable object is added to the graph (via attribute assignment),
   the system checks if a deferred restoration exists for it.
6. If found, the object's values are restored immediately.

### create_if_missing

Variables can be created without initial values when restoring:

```python
# The _UninitializedVariable is created and immediately gets its value
# from the checkpoint
class _UninitializedVariable(variables.Variable):
    # Variable created without running the initializer
    pass
```

### Limitations

- The object graph structure (attribute names, hierarchy) must match between
  save and restore.
- Variable dtypes and shapes must be compatible.
- Slot variables require both the optimizer and the original variable to be
  tracked.

---

## Training State

### Saving Optimizer State

Optimizers are fully trackable and their state (momentum, Adam statistics, etc.)
is automatically saved:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
checkpoint = tf.train.Checkpoint(
    model=model,
    optimizer=optimizer,
    step=tf.Variable(0, dtype=tf.int64)
)

# The optimizer's slot variables (m, v for Adam) are saved alongside
# the model variables
path = checkpoint.save('/tmp/ckpt')
```

### Global Step

The global step is commonly tracked as a variable:

```python
step = tf.Variable(0, name='global_step', dtype=tf.int64)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)

# In training loop:
for batch in dataset:
    train_step(batch)
    step.assign_add(1)
    if step % 1000 == 0:
        checkpoint.save('/tmp/ckpt')
```

### Slot Variables

Optimizer slot variables (e.g., Adam's first and second moment estimates) are
saved with references to both the optimizer and the original variable:

```protobuf
message SlotVariableReference {
  string slot_name = 1;  // e.g., "m" or "v" for Adam
  int32 original_variable_node_id = 2;
  int32 slot_variable_node_id = 3;
}
```

This ensures that slot variables are correctly matched during restoration even
if the variable ordering changes.

### Iterators

`tf.data.Dataset` iterators can be saved to preserve training progress:

```python
dataset = tf.data.Dataset.range(100).batch(10)
iterator = iter(dataset)
checkpoint = tf.train.Checkpoint(iterator=iterator)

# Save and restore the iterator position
checkpoint.save('/tmp/ckpt')
# ... later ...
checkpoint.restore('/tmp/ckpt')
# Iterator resumes from where it left off
```

---

## Checkpoints in Distributed Training

### Sharding

In distributed settings, variables may reside on different devices. Each worker
writes its own section of the checkpoint:

```python
# Automatic sharding by device
options = tf.train.CheckpointOptions(
    experimental_io_device='/job:localhost')
checkpoint.save('/tmp/ckpt', options=options)
```

### experimental_io_device

The `experimental_io_device` option controls which device is used for I/O
operations during checkpointing:

- `None` (default): Each variable's I/O runs on the CPU of its assigned host.
- `'/job:localhost'`: All I/O runs on the local host. Useful for saving to
  local directories in distributed settings.
- `'/job:worker/task:0'`: All I/O runs on a specific worker.

```python
options = tf.train.CheckpointOptions(
    experimental_io_device='/job:localhost')

# Save
checkpoint.save('/tmp/ckpt', options=options)

# Restore
checkpoint.restore('/tmp/ckpt', options=options)
```

### MirroredVariables

When using `tf.distribute.MirroredStrategy`, variables are replicated across
devices. By default, only one copy is saved. With
`EXPAND_DISTRIBUTED_VARIABLES` policy, all replicas are saved:

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
    optimizer = create_optimizer()

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
# Variables are saved as a single copy by default
```

### AsyncCheckpoint

Asynchronous checkpointing offloads the save operation to a background thread,
reducing training interruptions:

```python
options = tf.train.CheckpointOptions(
    experimental_enable_async_checkpoint=True)
checkpoint.save('/tmp/ckpt', options=options)
```

The `AsyncCheckpointHelper` class manages the background thread:
1. Copies variable values to CPU tensors.
2. Dispatches the serialization to a background thread.
3. The main thread continues training immediately.
4. The background thread writes the checkpoint.

To ensure the checkpoint is fully written before program exit:
```python
checkpoint.sync()
```

---

## TF1 Saver Compatibility

### tf.train.Saver

The TF1 `tf.train.Saver` uses name-based checkpointing:

```python
# TF1 style
saver = tf.compat.v1.train.Saver(var_list)
saver.save(sess, '/tmp/model.ckpt')
saver.restore(sess, '/tmp/model.ckpt')
```

### Loading TF1 Checkpoints with tf.train.Checkpoint

`tf.train.Checkpoint.restore()` can load name-based checkpoints:

```python
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore('/tmp/tf1_model.ckpt')
# Uses variable names to match
```

When loading name-based checkpoints:
- Variables are matched by their global name (e.g., `"dense/kernel:0"`).
- No deferred restoration is supported.
- No restore-on-create; variables must exist before restoration.
- `assert_consumed()` checks all variable names match.

### tf.train.latest_checkpoint

```python
# Works with both TF1 and TF2 checkpoints
latest = tf.train.latest_checkpoint('/tmp/checkpoint_dir')
# Returns: '/tmp/checkpoint_dir/ckpt-5' or None
```

This function reads the `checkpoint` state file and returns the most recent
checkpoint path.

### get_checkpoint_state

```python
state = tf.train.get_checkpoint_state('/tmp/checkpoint_dir')
print(state.model_checkpoint_path)
print(state.all_model_checkpoint_paths)
```

### checkpoint_exists

```python
exists = tf.train.checkpoint_exists('/tmp/model.ckpt')
```

---

## Checkpoint Conversion

### V1-to-V2 Conversion

To convert a name-based (TF1) checkpoint to an object-based (TF2) checkpoint:

```python
# 1. Create the same model
model = create_model()

# 2. Create a Checkpoint
checkpoint = tf.train.Checkpoint(model=model)

# 3. Restore from the TF1 checkpoint (name-based matching)
checkpoint.restore('/tmp/v1_model.ckpt')

# 4. Save as a TF2 checkpoint (object-based)
checkpoint.save('/tmp/v2_model')
```

### V2-to-V1 Conversion

To convert an object-based checkpoint to a name-based one:

```python
# 1. Load the TF2 checkpoint
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore('/tmp/v2_model-1')

# 2. Create a TF1 Saver with the variables
saver = tf.compat.v1.train.Saver(var_list=model.variables)

# 3. Save as a TF1 checkpoint
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # Set variable values from the loaded checkpoint
    saver.save(sess, '/tmp/v1_model.ckpt')
```

---

## Best Practices

### Saving Frequency

- Save at regular intervals (e.g., every N steps or every epoch).
- Use `CheckpointManager` with `max_to_keep` to limit disk usage.
- Use `keep_checkpoint_every_n_hours` for time-based preservation.

```python
manager = tf.train.CheckpointManager(
    checkpoint,
    directory='/tmp/training',
    max_to_keep=5,
    keep_checkpoint_every_n_hours=4
)
```

### Cleanup

- Use `max_to_keep` to automatically delete old checkpoints.
- Periodically verify that checkpoints can be restored.
- Monitor disk usage in long-running training jobs.

### Backup

- Copy important checkpoints to backup storage.
- Use `keep_checkpoint_every_n_hours` to ensure periodic snapshots.
- Consider saving to multiple locations.

### Error Handling

```python
try:
    status = checkpoint.restore(manager.latest_checkpoint)
    status.assert_existing_objects_matched()
except tf.errors.NotFoundError:
    print("Checkpoint not found, starting from scratch")
```

### Checkpoint Validation

```python
# Verify checkpoint integrity
import tensorflow as tf
reader = tf.train.load_checkpoint('/tmp/ckpt-1')
print(reader.get_variable_to_shape_map())
print(reader.get_variable_to_dtype_map())
```

### Naming Conventions

- Use descriptive prefixes: `/models/experiment_001/ckpt`
- Include version or date information.
- Use consistent directory structure across experiments.

---

## Warm Starting

### tf.estimator.WarmStartSettings

Warm starting allows initializing a model from a previously trained model's
weights, even if the architecture has changed:

```python
warm_start = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from='/tmp/pretrained_model',
    vars_to_warm_start='.*',  # Regex pattern for variables to warm start
    var_name_to_vocab_info={
        'input_layer/embedding_weights': tf.estimator.VocabInfo(
            new_vocab='new_vocab.txt',
            new_vocab_size=10000,
            num_oov_buckets=1,
            old_vocab='old_vocab.txt',
            old_vocab_size=5000,
            backup_initializer=tf.keras.initializers.TruncatedNormal())
    }
)

estimator = tf.estimator.DNNClassifier(
    hidden_units=[128, 64],
    feature_columns=feature_columns,
    warm_start_from=warm_start
)
```

### warm_starting_util

The `warm_starting_util` module provides low-level utilities:

```python
from tensorflow.python.training import warm_starting_util

# Warm start specific variables
warm_starting_util.warm_start(
    ckpt_to_initialize_from='/tmp/model',
    vars_to_warm_start=['dense/kernel', 'dense/bias'],
    vocab_info={
        'embedding': VocabInfo(...)
    }
)
```

### Partial Warm Starting

```python
# Only warm start the backbone
warm_start = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from='/tmp/pretrained',
    vars_to_warm_start='backbone/.*'  # Regex for backbone variables
)
```

---

## Internal Architecture

### TrackableSaver

The `TrackableSaver` class (in `tensorflow.python.checkpoint.checkpoint`)
handles the actual save/restore operations:

```python
class TrackableSaver:
    def __init__(self, graph_view):
        self._graph_view = graph_view
        self._cache = None  # Cache for graph building
        self._saveables_cache = None

    def save(self, file_prefix, checkpoint_number=None, session=None, options=None):
        # 1. Gather serialized tensors from all trackable objects
        # 2. Create or retrieve cached save operations
        # 3. Execute save (eagerly or via session)
        ...

    def restore(self, save_path, options=None):
        # 1. Read checkpoint reader
        # 2. Parse TrackableObjectGraph
        # 3. Create CheckpointRestoreCoordinator
        # 4. Match objects and restore values
        ...
```

### ObjectGraphView

The `ObjectGraphView` class (in `tensorflow.python.checkpoint.graph_view`)
traverses the object graph:

```python
class ObjectGraphView:
    def __init__(self, root, attached_dependencies=None):
        self.root = root
        self.attached_dependencies = attached_dependencies or []

    def list_children(self, obj, save_type):
        # Enumerate tracked children of obj
        ...
```

### CheckpointPosition

The `CheckpointPosition` class (in `tensorflow.python.checkpoint.restore`)
represents a single node in the checkpoint's object graph and handles the
restoration of its values:

```python
class CheckpointPosition:
    def __init__(self, checkpoint, proto_id):
        self.checkpoint = checkpoint
        self.proto_id = proto_id

    def restore(self, trackable, reader):
        # Match this checkpoint position to a Python object
        # Restore variable values
        # Handle deferred dependencies
        ...
```

### CheckpointRestoreCoordinator

Manages the state of an ongoing checkpoint restoration:

```python
class _CheckpointRestoreCoordinator:
    def __init__(self, object_graph_proto, save_path, save_path_tensor,
                 reader, restore_op_cache, graph_view, options, saveables_cache):
        self.object_graph_proto = object_graph_proto
        self.object_by_proto_id = weakref.WeakValueDictionary()
        self.matched_proto_ids = set()
        self.slot_restorations = collections.defaultdict(list)
        self.deferred_slot_restorations = {}
        ...
```

### Save Utilities

The `save_util` module provides helper functions:
- `serialize_graph_view()`: Serializes all objects in the graph to tensors.
- `objects_ids_and_slot_variables_and_paths()`: Enumerates all objects,
  assigns node IDs, and finds slot variables.

### Functional Saver

The `functional_saver` module implements the actual checkpoint writing:
- `MultiDeviceSaver`: Handles saving variables across multiple devices.
- Creates save/restore ops for each device.
- Supports sharded output.

---

## CheckpointOptions

```python
tf.train.CheckpointOptions(
    experimental_io_device=None,
    experimental_enable_async_checkpoint=False,
    experimental_write_callbacks=None,
    experimental_sharding_callback=None
)
```

### Parameters

**experimental_io_device** (`str` or `None`):
TensorFlow device for filesystem operations. See the distributed training
section for details.

**experimental_enable_async_checkpoint** (`bool`):
If `True`, enables asynchronous checkpoint saving. The save operation copies
variable values to CPU tensors and dispatches the actual I/O to a background
thread.

**experimental_write_callbacks** (`list[callable]` or `None`):
List of callback functions executed after the checkpoint is written. Each
callback can accept 0 or 1 parameters (the save path):
```python
def my_callback(save_path):
    print(f"Checkpoint saved to {save_path}")

options = tf.train.CheckpointOptions(
    experimental_write_callbacks=[my_callback])
```

**experimental_sharding_callback** (`ShardingCallback` or `None`):
Controls how checkpoint data files are sharded.

### ShardingCallback

Pre-made sharding policies:

```python
# Shard by device
options = tf.train.CheckpointOptions(
    experimental_sharding_callback=tf.train.experimental.ShardByDevicePolicy())

# Shard by maximum size
options = tf.train.CheckpointOptions(
    experimental_sharding_callback=tf.train.experimental.MaxShardSizePolicy(
        max_shard_size_bytes=100_000_000))  # 100MB per shard
```

Custom sharding:
```python
class MyShardingCallback(tf.train.experimental.ShardingCallback):
    def __call__(self, saveables):
        # Return a mapping from shard names to saveable objects
        ...
```
