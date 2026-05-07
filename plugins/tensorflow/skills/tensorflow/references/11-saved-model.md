# TensorFlow SavedModel Reference

## Table of Contents

1. [Overview](#overview)
2. [SavedModel Format](#savedmodel-format)
3. [tf.saved_model.save()](#tfsaved_modelsave)
4. [tf.saved_model.load()](#tfsaved_modelload)
5. [SaveOptions](#saveoptions)
6. [LoadOptions](#loadoptions)
7. [Signatures](#signatures)
8. [ConcreteFunction](#concretefunction)
9. [Keras Model Saving](#keras-model-saving)
10. [Loading Keras Models](#loading-keras-models)
11. [Version Compatibility](#version-compatibility)
12. [SavedModel Fingerprinting](#savedmodel-fingerprinting)
13. [MetaGraphDef](#metagraphdef)
14. [Exporting for Serving](#exporting-for-serving)
15. [Multi-Signature Models](#multi-signature-models)
16. [SavedModel with TFLite](#savedmodel-with-tflite)
17. [Large Model Support](#large-model-support)
18. [Security Considerations](#security-considerations)
19. [TF1 SavedModel Compatibility](#tf1-savedmodel-compatibility)
20. [Internal Architecture](#internal-architecture)

---

## Overview

SavedModel is TensorFlow's standard serialization format for complete models,
including their computation graphs, variable values, and signatures (input/output
specifications). It provides a language-neutral, recoverable, and hermetic
serialization format that enables higher-level systems and tools to produce,
consume, and transform TensorFlow models.

The SavedModel format was designed to be the primary mechanism for exporting
trained models for deployment. It encapsulates everything needed to run a model:
the computation graph (or `tf.function` objects), trained variable values, asset
files (such as vocabulary files), and signature definitions that describe how to
invoke the model.

Key design principles:
- **Hermetic**: A SavedModel directory contains all necessary files and does not
  depend on external code or files beyond the TensorFlow runtime.
- **Recoverable**: The directory structure is well-defined and can be validated.
- **Language-neutral**: The format uses protocol buffers for metadata, allowing
  consumption from Python, C++, Java, Go, and other TensorFlow-supported languages.
- **Version-aware**: Producer/consumer version metadata enables forward and
  backward compatibility checks.

---

## SavedModel Format

### Directory Structure

A SavedModel is stored as a directory with the following structure:

```
saved_model_dir/
    saved_model.pb              # The SavedModel protocol buffer (binary)
    fingerprint.pb              # Fingerprint hash of the SavedModel (optional)
    variables/
        variables.index         # Index file for checkpoint variables
        variables.data-00000-of-00001  # Data file(s) for variable values
    assets/                     # Directory for external files (vocabularies, etc.)
        asset_file.txt
    assets.extra/               # Additional assets directory (optional)
    debug/                      # Debug information (optional)
        saved_model_debug_info.pb
```

### saved_model.pb

The `saved_model.pb` file is the core protocol buffer that contains:

- **SavedModel proto**: The top-level container with schema version.
- **MetaGraphDef**: One or more meta graphs, each containing:
  - GraphDef: The computation graph (nodes, operations, functions).
  - SignatureDef: Named input/output specifications for serving.
  - CollectionDef: Named collections of tensors or operations.
  - SaverDef: Information about how to save/restore variables.
  - ObjectGraphDef (TF2): Serialized object graph for Python object reconstruction.
- **AssetFileDef**: References to external asset files.

The `SavedModel` proto definition:

```protobuf
message SavedModel {
  int64 saved_model_schema_version = 1;
  repeated MetaGraphDef meta_graphs = 2;
}
```

The schema version is set to `SAVED_MODEL_SCHEMA_VERSION` (currently 1).

### variables/

The `variables/` subdirectory contains checkpoint files in the standard TensorFlow
checkpoint format:

- **variables.index**: An index file mapping variable names to their locations
  in the data files. Also contains the `TrackableObjectGraph` proto that encodes
  the object dependency graph.
- **variables.data-NNNNN-of-MMMMM**: One or more data files containing the actual
  serialized variable tensor values. These can be sharded across multiple files
  for large models.

### assets/

The `assets/` directory contains external files that the model depends on, such as:
- Vocabulary files for text processing
- Label mapping files
- Configuration files
- Any other data files referenced by the graph

Each asset is tracked by an `AssetFileDef` protocol buffer in the `saved_model.pb`.

### fingerprint.pb

A relatively recent addition, the `fingerprint.pb` file contains a
`FingerprintDef` protocol buffer with cryptographic hashes of the SavedModel's
components. This enables quick comparison and verification of SavedModels without
parsing the entire model.

```protobuf
message FingerprintDef {
  uint64 saved_model_checksum = 1;
  uint64 graph_def_program_hash = 2;
  uint64 signature_def_hash = 3;
  uint64 saved_object_graph_hash = 4;
  uint64 checkpoint_hash = 5;
  int32 version = 6;
}
```

---

## tf.saved_model.save()

### Signature

```python
tf.saved_model.save(
    obj,
    export_dir,
    signatures=None,
    options=None
)
```

### Parameters

**obj** (`Trackable`):
A trackable object to export. This must inherit from `Trackable` (e.g.,
`tf.Module`, `tf.keras.Model`, `tf.train.Checkpoint`, or their subclasses).
The object and all its transitive dependencies (variables, sub-modules,
functions) are saved.

**export_dir** (`str` or `PathLike`):
The directory path where the SavedModel will be written. The directory is
created if it does not exist. If the directory already exists, it will be
overwritten.

**signatures** (optional):
Controls which methods are available to consumers. Accepts three forms:

1. **A `tf.function` with input_signature**: Used as the default serving
   signature.
   ```python
   @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
   def serve(self, x):
       return self.model(x)
   tf.saved_model.save(module, path, signatures=module.serve)
   ```

2. **A ConcreteFunction**: The result of `f.get_concrete_function(...)` on a
   `@tf.function`-decorated method.
   ```python
   tf.saved_model.save(
       module, path,
       signatures=module.serve.get_concrete_function(
           tf.TensorSpec([], tf.float32)))
   ```

3. **A dictionary**: Maps signature keys to `tf.function` instances or concrete
   functions.
   ```python
   tf.saved_model.save(
       module, path,
       signatures={
           'serving_default': module.serve,
           'classify': module.classify.get_concrete_function(...)
       })
   ```

If `signatures` is omitted, the system searches `obj` for `@tf.function`-decorated
methods. If exactly one traced `@tf.function` is found, it becomes the default
signature. Otherwise, all `@tf.function` methods are exported but only accessible
via `tf.saved_model.load`.

**options** (`tf.saved_model.SaveOptions`):
Configuration options for the save operation. See the SaveOptions section below.

### Behavior Details

1. **Object Graph Serialization**: The save process traverses the object graph
   starting from `obj`, collecting all trackable dependencies (variables, layers,
   optimizers, functions, assets). Each object is assigned a node ID in the
   `SavedObjectGraph` protocol buffer.

2. **Function Serialization**: Each `@tf.function` and its concrete functions are
   serialized as `FunctionDef` protocol buffers in the `GraphDef`'s function
   library. Concrete functions with cached variable captures are unwrapped to
   ensure proper variable restoration.

3. **Checkpoint Writing**: Variable values are saved to the `variables/`
   subdirectory using the standard checkpoint mechanism. The checkpoint includes
   the `TrackableObjectGraph` proto.

4. **Asset Copying**: Referenced asset files are copied into the `assets/`
   directory within the SavedModel.

5. **Atomic Write**: The `saved_model.pb` file is written atomically as the last
   file operation. This ensures that checking for its existence is a reliable
   way to determine if a SavedModel has been completely written.

6. **Debug Stripping**: When `experimental_debug_stripper=True`, `Assert` and
   `CheckNumerics` nodes are removed from the exported graph to reduce overhead
   during inference.

### Error Conditions

- `ValueError`: If `obj` is not trackable.
- `AssertionError`: If called from within a `@tf.function` (must be called in
  eager context).
- `ValueError`: If signature argument names are not unique.
- `ValueError`: If there is a cyclic dependency in the object graph.

### Example: Basic Save

```python
class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.v = tf.Variable(2.0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def __call__(self, x):
        return self.v * x

module = MyModule()
tf.saved_model.save(module, '/tmp/my_module')
```

### Example: Multi-Signature Save

```python
class MultiModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.v = tf.Variable(1.0)

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
    def serve(self, x):
        return x @ tf.eye(3) * self.v

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
    def classify(self, x):
        return tf.nn.softmax(x * self.v)

model = MultiModel()
tf.saved_model.save(
    model, '/tmp/multi',
    signatures={
        'serving_default': model.serve,
        'classify': model.classify
    })
```

---

## tf.saved_model.load()

### Signature

```python
tf.saved_model.load(
    export_dir,
    tags=None,
    options=None
)
```

### Parameters

**export_dir** (`str` or `PathLike`):
The directory path of the SavedModel to load.

**tags** (`str`, list of `str`, or `set`):
Tags identifying which MetaGraph to load. Optional if the SavedModel contains
a single MetaGraph (as for those exported from `tf.saved_model.save`). Typical
values come from `tf.saved_model.tag_constants`:
- `SERVING` ("serve")
- `TRAINING` ("train")
- `GPU` ("gpu")
- `TPU` ("tpu")

**options** (`tf.saved_model.LoadOptions`):
Configuration options for loading. See the LoadOptions section below.

### Return Value

Returns a trackable object with:
- **`.signatures`**: A dictionary mapping signature keys to callable concrete
  functions.
- **Trackable attributes**: If saved with `tf.saved_model.save`, the loaded
  object has attributes corresponding to the saved object's tracked dependencies.
- **`.tensorflow_version`**: The TensorFlow version used to create the SavedModel.
- **`.tensorflow_git_version`**: The Git version of TensorFlow used to save.

### Behavior Details

1. **Proto Parsing**: The loader reads `saved_model.pb` and parses the
   `SavedModel` protocol buffer.

2. **Object Graph Reconstruction**: For TF2 SavedModels, the `SavedObjectGraph`
   is deserialized to reconstruct Python objects. Each node type is handled
   differently:
   - **Variables**: Recreated as `tf.Variable` with saved shape, dtype, and
     device information.
   - **Functions**: Recreated as restored `tf.function` objects with their
     concrete functions.
   - **Assets**: Recreated as `Asset` objects referencing files in the `assets/`
     directory.
   - **User Objects**: Recreated based on registered type deserializers.
   - **Resources**: Recreated as `RestoredResource` objects.

3. **Checkpoint Restoration**: Variable values are restored from the `variables/`
   subdirectory using the object-based checkpoint restoration mechanism.

4. **Function Deserialization**: Concrete functions are loaded from the
   `FunctionDefLibrary` in the `GraphDef`, with captures bound to the
   reconstructed objects.

5. **Fingerprint Validation**: The fingerprint is read and validated against
   the SavedModel contents.

### Loading TF2 SavedModels

```python
imported = tf.saved_model.load('/tmp/my_module')
# Access variables
print(imported.v.numpy())  # 2.0
# Call the function directly (if __call__ was exported)
result = imported(tf.constant(3.0))  # 6.0
# Access signatures
f = imported.signatures['serving_default']
result = f(x=tf.constant(3.0))
```

### Loading TF1 SavedModels

TF1 SavedModels are loaded with additional attributes:
- **`.signatures`**: Dictionary of signature names to functions.
- **`.prune(feeds, fetches)`**: Extract functions for new subgraphs.
- **`.variables`**: List of imported variables.
- **`.graph`**: The imported graph.
- **`.restore(save_path)`**: Restore variables from a TF1 checkpoint.

```python
imported = tf.saved_model.load('/tmp/v1_model')
pruned = imported.prune("x:0", "out:0")
result = pruned(tf.ones([]))
```

### Async Consumption

When consuming SavedModels asynchronously (producer is a separate process),
check for `saved_model_dir/saved_model.pb` rather than the directory itself,
since this file is written atomically as the last operation.

---

## SaveOptions

```python
tf.saved_model.SaveOptions(
    namespace_whitelist=None,
    save_debug_info=False,
    function_aliases=None,
    experimental_debug_stripper=False,
    experimental_io_device=None,
    experimental_variable_policy=None,
    experimental_custom_gradients=True,
    experimental_image_format=False,
    experimental_skip_saver=False,
    experimental_sharding_callback=None,
    extra_tags=None
)
```

### Parameters

**namespace_whitelist** (`list[str]` or `None`):
Op namespaces to whitelist when saving. If `None` (default), all namespaced
ops are allowed. If a list, only ops in the listed namespaces are permitted.
This ensures that all custom ops used in the model will be available at load time.

```python
options = tf.saved_model.SaveOptions(
    namespace_whitelist=['my_custom_ops'])
```

**save_debug_info** (`bool`):
If `True`, writes a `debug/saved_model_debug_info.pb` file containing
`GraphDebugInfo` with stack trace information for all ops and functions.

**function_aliases** (`dict[str, object]`):
Mapping from string alias names to `@tf.function` objects. Since a single
`tf.function` can generate many `ConcreteFunction`s, this allows downstream
tools to refer to all concrete functions generated by a single `tf.function`
using a single alias.

```python
options = tf.saved_model.SaveOptions(
    function_aliases={'double': model.double})
```

**experimental_debug_stripper** (`bool`):
If `True`, strips `Assert` and `CheckNumerics` nodes from the exported graph.
Assert nodes become `NoOp`s, CheckNumerics nodes become `Identity` ops.

**experimental_io_device** (`str` or `None`):
In distributed settings, the TensorFlow device to use for filesystem access.
If `None` (default), each variable's filesystem is accessed from the CPU:0
device of the host where that variable is assigned. Useful for saving to local
directories in distributed settings:
```python
options = tf.saved_model.SaveOptions(
    experimental_io_device='/job:localhost')
```

**experimental_variable_policy** (`VariablePolicy` or `str`):
Controls how variables are handled during saving. Options:
- `NONE` (default): Distributed variables are saved as one variable, no device
  attached.
- `SAVE_VARIABLE_DEVICES`: Saves variable device assignments. Useful for
  hardcoding devices but makes models non-portable.
- `EXPAND_DISTRIBUTED_VARIABLES`: Saves component information of distributed
  variables, enabling restoration without the original distribution strategy.

**experimental_custom_gradients** (`bool`):
When `True` (default), saves traced gradient functions for functions decorated
with `tf.custom_gradient`. Disabling this can reduce SavedModel size.

**experimental_image_format** (`bool`):
Enables a new format capable of saving models larger than the 2GB protobuf
limit. Currently disabled in OSS builds.

**experimental_skip_saver** (`bool`):
If `True`, prevents creation of native checkpoint ops. Useful for models that
do not use SavedModel's checkpointing functionality.

**experimental_sharding_callback** (`ShardingCallback`):
Determines how checkpoint files are sharded on disk. Pre-made callbacks include
`ShardByDevicePolicy` and `MaxShardSizePolicy`.

**extra_tags** (`list[str]` or `None`):
Extra tags to save with the MetaGraph in addition to the default "serve" tag.

---

## LoadOptions

```python
tf.saved_model.LoadOptions(
    allow_partial_checkpoint=False,
    experimental_io_device=None,
    experimental_skip_checkpoint=False,
    experimental_variable_policy=None,
    experimental_load_function_aliases=False
)
```

### Parameters

**allow_partial_checkpoint** (`bool`):
When `True`, allows the SavedModel checkpoint to not entirely match the loaded
object. Useful when loading Keras models with custom objects that have gained
new variables since the SavedModel was created.

```python
# Loading a model where Custom.w was added after saving
options = tf.saved_model.LoadOptions(allow_partial_checkpoint=True)
tf.keras.models.load_model(path, custom_objects={'Custom': Custom}, options=options)
```

**experimental_io_device** (`str` or `None`):
TensorFlow device for filesystem access during loading. Same semantics as
`SaveOptions.experimental_io_device`.

**experimental_skip_checkpoint** (`bool`):
If `True`, checkpoints will not be restored. This will typically produce an
unusable model but can be useful for inspecting the graph structure.

**experimental_variable_policy** (`VariablePolicy` or `str`):
The policy to apply to variables when loading. Same options as
`SaveOptions.experimental_variable_policy`.

**experimental_load_function_aliases** (`bool`):
If `True`, adds a `function_aliases` attribute to the loaded object, mapping
alias names to the restored functions.

---

## Signatures

### Overview

Signatures define the input and output types for a computation. They are the
primary interface for consuming SavedModels, especially in serving contexts.

### SignatureDef Protocol Buffer

Each signature is encoded as a `SignatureDef` protocol buffer:

```protobuf
message SignatureDef {
  map<string, TensorInfo> inputs = 1;
  map<string, TensorInfo> outputs = 2;
  string method_name = 3;
}
```

Where `TensorInfo` specifies:
```protobuf
message TensorInfo {
  oneof encoding {
    string name = 1;          // Tensor name in the graph
    CooSparse coo_sparse = 2; // Sparse tensor encoding
  }
  DataType dtype = 3;
  TensorShapeProto tensor_shape = 4;
}
```

### Default Signature Key

The default serving signature uses the key `"serving_default"`, defined in
`tf.saved_model.signature_constants`:
- `DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"`
- `PREDICT_METHOD_NAME = "tensorflow/serving/predict"`
- `CLASSIFY_METHOD_NAME = "tensorflow/serving/classify"`
- `REGRESS_METHOD_NAME = "tensorflow/serving/regress"`

### Structured Signatures

Input and output names are derived from Python function argument names by default.
They can be overridden with `tf.TensorSpec(..., name="custom_name")`:

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32, name="images")
])
def serve(self, images):
    return {"predictions": self.model(images)}
```

Outputs can be either flat lists (numbered: `"output_0"`, `"output_1"`, etc.)
or dictionaries (using the dictionary keys as output names).

### Custom Signatures

Multiple signatures can be exported for different use cases:

```python
class Model(tf.Module):
    @tf.function
    def serve(self, x):
        return self.inference(x)

    @tf.function
    def train(self, x, y):
        return self.loss(x, y)

m = Model()
tf.saved_model.save(m, path, signatures={
    'serving_default': m.serve.get_concrete_function(
        tf.TensorSpec([None, 224, 224, 3], tf.float32)),
    'train': m.train.get_concrete_function(
        tf.TensorSpec([None, 224, 224, 3], tf.float32),
        tf.TensorSpec([None, 1000], tf.float32))
})
```

### Accessing Signatures After Loading

```python
imported = tf.saved_model.load(path)
# List available signatures
print(list(imported.signatures.keys()))  # ['serving_default']
# Call a signature
f = imported.signatures['serving_default']
result = f(images=tf.constant(...))
```

---

## ConcreteFunction

### Overview

A `ConcreteFunction` is a graph-backed callable that represents a single
computation graph with fixed input shapes and dtypes. It is the fundamental
unit of execution in SavedModels.

### Creation

ConcreteFunctions are created when:
1. Calling `get_concrete_function()` on a `@tf.function`:
   ```python
   @tf.function
   def f(x):
       return x * 2

   cf = f.get_concrete_function(tf.TensorSpec([None, 3], tf.float32))
   ```

2. Providing `input_signature` to `@tf.function`:
   ```python
   @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
   def g(x):
       return x + 1
   # g is automatically a ConcreteFunction
   ```

### Properties

- **`.graph`**: The `FuncGraph` containing the computation graph.
- **`.function_def`**: The `FunctionDef` protocol buffer.
- **`.name`**: The unique function name (e.g., `"my_function_123"`).
- **`.inputs`**: List of input tensors (including captured tensors).
- **`.outputs`**: List of output tensors.
- **`.structured_input_signature`**: The nested structure of input specs.
- **`.structured_outputs`**: The nested structure of outputs.
- **`.graph.captures`**: List of (external, internal) tensor pairs captured
  from the outer scope.

### Input/Output Specs

```python
cf = f.get_concrete_function(tf.TensorSpec([None, 3], tf.float32))

# Input specification
print(cf.structured_input_signature)
# ({'x': TensorSpec(shape=(None, 3), dtype=tf.float32, name='x')},)

# Output specification
print(cf.structured_outputs)
# TensorSpec(shape=(None, 3), dtype=tf.float32)
```

### Calling

ConcreteFunctions can be called directly with tensors:
```python
result = cf(x=tf.constant([[1.0, 2.0, 3.0]]))
```

---

## Keras Model Saving

### model.save() with SavedModel Format

Keras models can be saved using the `.save()` method, which internally uses
`tf.saved_model.save`:

```python
model = tf.keras.Model(inputs=x, outputs=y)
model.save('/tmp/my_model')  # SavedModel format by default
```

The `.keras` format (default in modern TensorFlow) stores additional metadata
including the model architecture, training configuration, and optimizer state:

```python
model.save('/tmp/my_model.keras')  # .keras format
model.save('/tmp/my_model.h5')     # HDF5 format (legacy)
model.save('/tmp/my_model', save_format='tf')  # Explicit SavedModel
```

### Automatic Signature Generation

Keras models constructed from inputs and outputs automatically have a serving
signature. The forward pass is exported:

```python
x = tf.keras.layers.Input((4,), name="x")
y = tf.keras.layers.Dense(5, name="out")(x)
model = tf.keras.Model(x, y)
tf.saved_model.save(model, '/tmp/model')
# The SavedModel takes "x" with shape [None, 4] and returns "out"
# with shape [None, 5]
```

### Saving Custom Layers

Custom layers must be properly tracked. Keras layers automatically track their
variables, but any additional state must be tracked manually:

```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.v = None

    def build(self, input_shape):
        self.v = self.add_weight(name='v', shape=input_shape[1:])

    def call(self, inputs):
        return inputs * self.v

    # get_config is needed for Keras format saving
    def get_config(self):
        return super().get_config()
```

---

## Loading Keras Models

### tf.keras.models.load_model()

```python
model = tf.keras.models.load_model('/tmp/my_model')
model = tf.keras.models.load_model('/tmp/my_model.keras')
```

### Custom Objects

When loading models with custom layers, functions, or other objects, they must
be provided via the `custom_objects` argument:

```python
model = tf.keras.models.load_model(
    '/tmp/my_model',
    custom_objects={
        'CustomLayer': CustomLayer,
        'custom_activation': custom_activation_fn
    })
```

Or by registering custom objects:
```python
@tf.keras.utils.register_keras_serializable()
class CustomLayer(tf.keras.layers.Layer):
    ...
```

### Loading with compile=False

If the training configuration cannot be reconstructed (e.g., custom optimizer),
load without compiling:

```python
model = tf.keras.models.load_model('/tmp/model', compile=False)
```

### Loading vs. tf.saved_model.load

| Feature | `tf.keras.models.load_model` | `tf.saved_model.load` |
|---------|------------------------------|----------------------|
| Returns Keras Model | Yes | No (AutoTrackable) |
| `.fit()`, `.predict()` | Yes | No |
| `.evaluate()` | Yes | No |
| Custom layers | Reconstructs | Not reconstructed |
| Model architecture | Reconstructed from metadata | Not available |
| Training config | Available (if saved) | Not available |

---

## Version Compatibility

### Producer/Consumer Versions

The SavedModel format uses a versioning system with `VersionDef`:

```protobuf
message VersionDef {
  int32 producer = 1;
  int32 min_consumer = 2;
  repeated int32 bad_consumers = 3;
}
```

- **producer**: The version of the code that produced the SavedModel.
- **min_consumer**: The minimum consumer version required.
- **bad_consumers**: Specific consumer versions known to be incompatible.

### Forward/Backward Compatibility

- **Forward compatibility**: Newer producers can create SavedModels that older
  consumers can read (within limits). Default-valued attributes are stripped
  to reduce incompatibilities.
- **Backward compatibility**: Newer consumers can read SavedModels from older
  producers.

### Attribute Stripping

SavedModels exported with `tf.saved_model.save` automatically strip
default-valued attributes, which removes one source of incompatibilities.
However, other sources remain:
- Operations not defined in the consumer's TensorFlow version.
- Changes in operation semantics.

### Minimum Consumer Version

The `MetaGraphDef.meta_info_def` contains:
- `tensorflow_version`: Full version string of the producing TensorFlow.
- `tensorflow_git_version`: Git commit hash of the producing TensorFlow.
- `stripped_default_attrs`: Whether default attributes were stripped.

---

## SavedModel Fingerprinting

### FingerprintDef Protocol Buffer

The fingerprint is stored in `fingerprint.pb` and contains:

```protobuf
message FingerprintDef {
  uint64 saved_model_checksum = 1;      // Hash of saved_model.pb
  uint64 graph_def_program_hash = 2;     // Hash of the graph structure
  uint64 signature_def_hash = 3;         // Hash of signature definitions
  uint64 saved_object_graph_hash = 4;    // Hash of object graph
  uint64 checkpoint_hash = 5;            // Hash of variable values
  int32 version = 6;                     // Fingerprint version
}
```

### FingerprintingHasher

The `FingerprintingHasher` computes deterministic hashes of each component:

1. **saved_model_checksum**: A hash of the serialized `saved_model.pb` file.
2. **graph_def_program_hash**: A hash of the `GraphDef` structure (ops, types,
   connectivity) excluding variable values and function bodies.
3. **signature_def_hash**: A hash of all signature definitions.
4. **saved_object_graph_hash**: A hash of the `SavedObjectGraph` proto. This
   also distinguishes TF2 from TF1 SavedModels (non-zero for TF2).
5. **checkpoint_hash**: A hash of the variable checkpoint data.

### tf.saved_model.experimental.Fingerprint

```python
@tf_export("saved_model.experimental.Fingerprint", v1=[])
class Fingerprint:
    saved_model_checksum: int
    graph_def_program_hash: int
    signature_def_hash: int
    saved_object_graph_hash: int
    checkpoint_hash: int
    version: int
```

### Reading Fingerprints

```python
# Read fingerprint from a SavedModel directory
fingerprint = tf.saved_model.experimental.read_fingerprint('/tmp/model')

# Access individual hash values
print(fingerprint.saved_model_checksum)
print(fingerprint.graph_def_program_hash)

# Get canonical identifier (singleprint)
sp = fingerprint.singleprint()
```

### Singleprint

The singleprint is a canonical identifier that uniquely identifies a SavedModel
based on the regularized fingerprint attributes (excluding `saved_model_checksum`
which is sensitive to immaterial changes). It is the string concatenation of
`graph_def_program_hash`, `signature_def_hash`, `saved_object_graph_hash`, and
`checkpoint_hash`, separated by `/`.

---

## MetaGraphDef

### Structure

The `MetaGraphDef` protocol buffer is the primary container within a SavedModel:

```protobuf
message MetaGraphDef {
  MetaInfoDef meta_info_def = 1;
  GraphDef graph_def = 2;
  SaverDef saver_def = 3;
  map<string, CollectionDef> collection_def = 4;
  map<string, SignatureDef> signature_def = 5;
  repeated AssetFileDef asset_file_def = 6;
  SavedObjectGraph object_graph_def = 7;
}
```

### MetaInfoDef

Contains metadata about the MetaGraph:

```protobuf
message MetaInfoDef {
  repeated string tags = 1;
  string tensorflow_version = 3;
  string tensorflow_git_version = 4;
  bool stripped_default_attrs = 5;
  OpList stripped_op_list = 6;
  map<string, string> function_aliases = 7;
}
```

### GraphDef

The computation graph containing:
- **NodeDef**: Individual operations with their attributes and connections.
- **FunctionDefLibrary**: Collection of function definitions (concrete functions,
  gradient functions).

### SaverDef

Contains checkpoint save/restore information:
```protobuf
message SaverDef {
  string filename_tensor_name = 1;
  string save_tensor_name = 2;
  string restore_op_name = 3;
  int32 max_to_keep = 4;
  bool sharded = 5;
  float keep_checkpoint_every_n_hours = 6;
  CheckpointFormatVersion version = 7;
}
```

### CollectionDef

Named collections of tensors or other values. Used primarily in TF1 for
organizing graph elements.

### ObjectGraphDef (TF2)

The `SavedObjectGraph` that encodes the Python object hierarchy for TF2
SavedModels. This enables reconstruction of Python objects on load.

---

## Exporting for Serving

### TF-Serving Signature Conventions

TensorFlow Serving expects specific signature conventions:

#### Predict API

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="inputs")
])
def serve(self, inputs):
    predictions = self.model(inputs)
    return {"predictions": predictions}
```

#### Classify API

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.string, name="inputs")
])
def classify(self, inputs):
    # Returns classes and scores
    return {
        "classes": tf.constant(["cat", "dog"]),
        "scores": self.model(process(inputs))
    }
```

#### Regress API

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.float32, name="inputs")
])
def regress(self, inputs):
    return {"outputs": self.model(inputs)}
```

### Exporting with Specific Tags

```python
tf.saved_model.save(
    model, export_dir,
    signatures={'serving_default': serve_fn},
    options=tf.saved_model.SaveOptions(extra_tags=['train']))
```

### Signature Constants

```python
from tensorflow.python.saved_model import signature_constants

DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"
CLASSIFY_INPUTS = "inputs"
CLASSIFY_OUTPUT_CLASSES = "classes"
CLASSIFY_OUTPUT_SCORES = "scores"
PREDICT_INPUTS = "inputs"
PREDICT_OUTPUTS = "outputs"
REGRESS_INPUTS = "inputs"
REGRESS_OUTPUTS = "outputs"
```

---

## Multi-Signature Models

### Creating Multiple Entry Points

```python
class ServingModule(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec([None, 224, 224, 3], tf.float32)
    ])
    def predict(self, images):
        return self.model(images, training=False)

    @tf.function(input_signature=[
        tf.TensorSpec([None, 224, 224, 3], tf.float32),
        tf.TensorSpec([None], tf.int32)
    ])
    def extract_features(self, images, layer_indices):
        return self.model.extract(images, layer_indices)

    @tf.function(input_signature=[
        tf.TensorSpec([None, 224, 224, 3], tf.float32),
        tf.TensorSpec([None, 10], tf.float32)
    ])
    def compute_loss(self, images, labels):
        logits = self.model(images, training=False)
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)

module = ServingModule(my_model)
tf.saved_model.save(module, export_dir, signatures={
    'serving_default': module.predict,
    'extract_features': module.extract_features,
    'compute_loss': module.compute_loss
})
```

### Consuming Multi-Signature Models

```python
loaded = tf.saved_model.load(export_dir)
predict_fn = loaded.signatures['serving_default']
features_fn = loaded.signatures['extract_features']
loss_fn = loaded.signatures['compute_loss']
```

---

## SavedModel with TFLite

### Conversion Path

```python
import tensorflow as tf

# Save a model in SavedModel format
model = tf.keras.Model(...)
tf.saved_model.save(model, '/tmp/model')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('/tmp/model')
tflite_model = converter.convert()

# Save the TFLite model
with open('/tmp/model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Conversion Options

```python
converter = tf.lite.TFLiteConverter.from_saved_model('/tmp/model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.experimental_new_converter = True
tflite_model = converter.convert()
```

### Signature-Based Conversion

When a SavedModel has multiple signatures, TFLite can target specific ones:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(
    '/tmp/model',
    signature_keys=['serving_default']
)
```

---

## Large Model Support

### Protobuf Size Limitation

Standard protocol buffers have a 2GB size limit. For models that exceed this
limit, TensorFlow provides sharding capabilities.

### Proto Splitter

The experimental image format (`experimental_image_format=True`) uses
`SavedModelSplitter` to split large protos:

```python
from tensorflow.python.saved_model import proto_splitter

splitter = proto_splitter.SavedModelSplitter(saved_model_proto)
splitter.write(prefix)
```

### Sharded Saving

The `experimental_sharding_callback` option controls how checkpoint data files
are sharded:

```python
options = tf.saved_model.SaveOptions(
    experimental_sharding_callback=
        tf.train.experimental.ShardByDevicePolicy()
)
tf.saved_model.save(model, path, options=options)
```

Available sharding policies:
- **ShardByDevicePolicy**: Creates one shard per device.
- **MaxShardSizePolicy**: Creates shards up to a maximum size.
- **Custom ShardingCallback**: User-defined sharding logic.

### Multi-Device Saver

The `MultiDeviceSaver` handles saving variables distributed across multiple
devices, creating separate shards for each device and merging them.

---

## Security Considerations

### Model Signing

SavedModels can be cryptographically signed to ensure authenticity and integrity.
The signature is stored alongside the model and can be verified before loading.

### Verification

Before loading a SavedModel in production:
1. Verify the fingerprint matches the expected value.
2. Check file integrity using the `fingerprint.pb` hash values.
3. Validate that the model was produced by a trusted source.

### Deserialization Risks

SavedModels contain serialized computation graphs that execute arbitrary
TensorFlow operations. Only load SavedModels from trusted sources.

### Best Practices

1. **Restrict namespace whitelist**: Only allow known custom op namespaces.
2. **Use debug stripper**: Remove assertion and debug nodes before deployment.
3. **Validate fingerprints**: Compare fingerprints against known-good values.
4. **File permissions**: Ensure SavedModel directories have appropriate
   read/write permissions.
5. **Atomic writes**: Rely on the atomic `saved_model.pb` write to detect
   incomplete saves.

---

## TF1 SavedModel Compatibility

### builder.py (TF1)

The TF1 `SavedModelBuilder` API:

```python
builder = tf.saved_model.Builder(export_dir)
# Add a MetaGraph with tags
with tf.compat.v1.Session() as sess:
    # ... build graph ...
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'serving_default': signature_def
        },
        assets_collection=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS)
    )
builder.save()
```

### loader.py (TF1)

The TF1 loader API:

```python
with tf.compat.v1.Session() as sess:
    tf.compat.v1.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        export_dir
    )
```

### tag_constants

Standard tag constants for MetaGraph selection:
```python
class TagConstants:
    SERVING = "serve"
    TRAINING = "train"
    GPU = "gpu"
    TPU = "tpu"
```

### Migration Path

1. **TF1 to TF2**: Use `tf.saved_model.load` which handles both TF1 and TF2
   SavedModels. TF1 SavedModels are loaded with the `.prune()` method for
   extracting subgraphs.

2. **V1-in-V2 loading**: The `load_v1_in_v2` module provides compatibility
   for loading TF1 SavedModels in TF2:
   ```python
   imported = tf.saved_model.load(path_to_v1_saved_model)
   pruned = imported.prune("x:0", "out:0")
   pruned(tf.ones([]))
   ```

3. **is_tf2_saved_model()**: Helper function to determine if a SavedModel
   uses TF2 semantics:
   ```python
   tf.saved_model.load.is_tf2_saved_model(export_dir)
   # Returns True if TF2, False if TF1
   ```

---

## Internal Architecture

### _AugmentedGraphView

Extends `ObjectGraphView` for SavedModel saving. It:
- Caches children and serialization state for consistent snapshots.
- Tracks wrapped functions that capture non-cached variables.
- Merges equivalent constant tensors and assets.
- Reports untraced functions to the user.

### _SaveableView

Provides a frozen view over a trackable root for the duration of the save:
- Collects all trackable objects and their node IDs.
- Generates save/restore functions for checkpoint compatibility.
- Manages concrete functions and gradient functions.
- Fills the `SavedObjectGraph` protocol buffer.

### Save Pipeline

1. **Input validation**: Check `obj` is trackable, not inside a function.
2. **Signature resolution**: Find or validate signatures.
3. **Canonicalization**: `canonicalize_signatures()` normalizes signature inputs.
4. **Graph view construction**: `_AugmentedGraphView` traverses the object graph.
5. **Saveable view**: `_SaveableView` freezes the view and generates functions.
6. **MetaGraph filling**: `_fill_meta_graph_def()` creates the exported graph.
7. **Object graph serialization**: `_serialize_object_graph()` creates the
   `SavedObjectGraph` proto.
8. **Checkpoint writing**: Variables saved to `variables/` directory.
9. **Asset copying**: Asset files copied to `assets/` directory.
10. **Proto writing**: `saved_model.pb` written atomically.
11. **Fingerprint writing**: `fingerprint.pb` written.

### Load Pipeline

1. **Proto parsing**: Read and parse `saved_model.pb`.
2. **Fingerprint reading**: Read and validate `fingerprint.pb`.
3. **TF1/TF2 detection**: Check for `object_graph_def` field.
4. **Object reconstruction**: `Loader._load_nodes()` creates Python objects.
5. **Function deserialization**: `load_function_def_library()` restores functions.
6. **Edge restoration**: `_load_edges()` reconnects object graph edges.
7. **Capture binding**: `_setup_function_captures()` binds function captures.
8. **Checkpoint restoration**: Variable values loaded from `variables/`.
9. **Resource initialization**: `CapturableResource` objects initialized.

### Registration System

The `registration` module allows custom types to register serializers and
deserializers for SavedModel:

```python
@tf.saved_model.experimental.register_serializable
class MyCustomType(tf.Module):
    def _serialize_to_proto(self, object_proto=None):
        # Serialize custom state
        ...

    @classmethod
    def _deserialize_from_proto(cls, proto, object_proto, dependencies, ...):
        # Reconstruct from proto
        ...
```

### Revived Types

The `revived_types` module handles deserialization of known types:
- Optimizers (identified by `"optimizer"`)
- Keras layers and models
- Custom registered types

Each type has a deserialization function that reconstructs the Python object
from the saved protocol buffer data.
