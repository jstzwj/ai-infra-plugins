# TensorFlow Distributed Training Reference

This document provides comprehensive reference documentation for all aspects of distributed training in TensorFlow, including all distribution strategies, cluster configuration, cross-device operations, distributed variables, and advanced patterns like DTensor.

---

## Table of Contents

1. [DistributionStrategy API](#distributionstrategy-api)
2. [MirroredStrategy](#mirroredstrategy)
3. [MultiWorkerMirroredStrategy](#multiworkermirroredstrategy)
4. [TPUStrategy](#tpustrategy)
5. [ParameterServerStrategy](#parameterserverstrategy)
6. [CentralStorageStrategy](#centralstoragestrategy)
7. [OneDeviceStrategy](#onedevicestrategy)
8. [ClusterResolver](#clusterresolver)
9. [Cross-Device Operations](#cross-device-operations)
10. [Distributed Variables](#distributed-variables)
11. [Input Distribution](#input-distribution)
12. [Custom Training Loops](#custom-training-loops-distributed)
13. [Keras Integration](#keras-integration)
14. [Multi-Worker Setup](#multi-worker-setup)
15. [Performance Tuning](#performance-tuning)
16. [Fault Tolerance](#fault-tolerance)
17. [DTensor](#dtensor)

---

## DistributionStrategy API

### Base Class: tf.distribute.Strategy

The base class for all distribution strategies. Provides the core API for distributed training.

```python
tf.distribute.Strategy
```

### Key Methods

#### scope()

Returns a context manager that sets this strategy as the current strategy.

```python
with strategy.scope():
    # Variables created here are distributed according to the strategy
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
```

**Important:** Model creation, optimizer creation, and any variable creation must happen inside `strategy.scope()`.

#### run()

Invokes `fn` on each replica with the provided arguments.

```python
per_replica_result = strategy.run(
    fn,                                # Function to run on each replica
    args=(),                           # Positional arguments
    kwargs=None,                       # Keyword arguments
    options=None                       # tf.distribute.RunOptions
)
```

#### reduce()

Reduces `value` across replicas.

```python
reduced_value = strategy.reduce(
    reduce_op,                         # tf.distribute.ReduceOp (SUM, MEAN, MAX, MIN)
    value,                             # Per-replica value to reduce
    axis=None                          # Axis to reduce along
)
```

**Usage:**
```python
# Sum losses across replicas
total_loss = strategy.reduce(
    tf.distribute.ReduceOp.SUM,
    per_replica_loss,
    axis=None
)

# Mean across replicas
mean_loss = strategy.reduce(
    tf.distribute.ReduceOp.MEAN,
    per_replica_loss,
    axis=None
)
```

#### experimental_local_results()

Returns the list of all local per-replica values.

```python
local_values = strategy.experimental_local_results(
    value                              # A value returned by run() or a distributed variable
)
```

#### num_replicas_in_sync

```python
print(strategy.num_replicas_in_sync)  # Number of replicas (e.g., number of GPUs)
```

### Utility Functions

```python
# Check if in a strategy scope
tf.distribute.has_strategy()            # Boolean
tf.distribute.get_strategy()            # Returns current strategy
tf.distribute.get_replica_context()     # Returns ReplicaContext or None

# Check context type
tf.distribute.in_cross_replica_context()  # Boolean

# Get number of replicas
tf.distribute.get_strategy().num_replicas_in_sync
```

### ReduceOp

```python
tf.distribute.ReduceOp
# Values:
#   SUM      - Sum across replicas
#   MEAN     - Mean across replicas
#   MAX      - Maximum across replicas
#   MIN      - Minimum across replicas
#   ONLY_FIRST_REPLICA - Return value from first replica only
```

---

## MirroredStrategy

Single-machine synchronous multi-GPU training. Each GPU has a copy of all model variables, and gradients are aggregated using all-reduce.

```python
tf.distribute.MirroredStrategy(
    devices=None,                      # List of device strings. E.g., ['/gpu:0', '/gpu:1']
                                       # None = use all available GPUs
    cross_device_ops=None              # Cross-device ops for gradient reduction
)
```

### How It Works

1. Variables are mirrored (replicated) on each GPU
2. Each replica computes forward and backward pass independently
3. Gradients are aggregated using all-reduce (NCCL by default)
4. The aggregated gradient is applied to all replicas identically

### Usage

```python
# Detect and use all available GPUs
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Specify specific GPUs
strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])

# With custom cross-device operations
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.NcclAllReduce()
)
```

### Full Training Example

```python
# 1. Create strategy
strategy = tf.distribute.MirroredStrategy()
print(f'Number of GPUs: {strategy.num_replicas_in_sync}')

# 2. Create model and optimizer inside scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# 3. Create dataset with global batch size
global_batch_size = 64 * strategy.num_replicas_in_sync
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

# 4. Train
model.fit(train_dataset, epochs=10)
```

### Cross-Device Operations for MirroredStrategy

```python
# NCCL all-reduce (default, most efficient for GPUs)
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.NcclAllReduce(
        num_packs=2,                   # Number of gradient packs for all-reduce
        shared_name=None
    )
)

# HierarchicalCopy (efficient for many GPUs)
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(
        num_packs=3
    )
)

# ReduceToOneDevice (simple, for debugging)
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.ReduceToOneDevice(
        reduce_device='/gpu:0',
        accumulation_fn=tf.math.add
    )
)

# CollectiveAllReduce (for multi-worker)
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.CollectiveAllReduce(
        group_size=None,
        group_key=None
    )
)
```

---

## MultiWorkerMirroredStrategy

Multi-machine synchronous training using collective all-reduce over the network.

```python
tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=None,             # ClusterResolver instance
    communication_options=None         # tf.distribute.experimental.CommunicationOptions
)
```

### CommunicationOptions

```python
tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.AUTO,
    # AUTO          - Let TF choose (default)
    # RING          - Ring-based all-reduce
    # NCCL          - NCCL all-reduce (GPU only, fastest)
    # AUTO_ALIASED  - Automatically select best option

    bytes_per_pack=0,                  # Integer. Bytes per gradient pack during all-reduce
    timeout_seconds=None,              # Float. Timeout for collective operations
    max_parallel_broadcasts=None,      # Max parallel broadcast operations
    max_parallel_allreduce=None,       # Max parallel all-reduce operations
    max_parallel_allgather=None,       # Max parallel all-gather operations
)
```

### Usage

```python
# Set up TF_CONFIG environment variable for each worker
# Worker 0 (chief):
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['host1:port1', 'host2:port2', 'host3:port3']
    },
    'task': {'type': 'worker', 'index': 0}
})

# Worker 1:
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['host1:port1', 'host2:port2', 'host3:port3']
    },
    'task': {'type': 'worker', 'index': 1}
})

# Create strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Use like MirroredStrategy
with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Global batch size = per_worker_batch * num_workers
global_batch_size = 64 * strategy.num_replicas_in_sync

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(10000).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

model.fit(dataset, epochs=10)
```

### Determining Chief Worker

```python
# The chief worker handles checkpointing and other coordination tasks
# With MultiWorkerMirroredStrategy, the first 'worker' (index 0) is the chief

# Check if this is the chief
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
is_chief = cluster_resolver.task_type == 'worker' and cluster_resolver.task_id == 0

# Only chief saves checkpoints
if is_chief:
    model.save('model.keras')
```

---

## TPUStrategy

Synchronous training on TPU devices.

```python
tf.distribute.TPUStrategy(
    cluster_resolver=None              # TPUClusterResolver instance
)
```

### Usage

```python
# Connect to TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu='grpc://' + os.environ['COLAB_TPU_ADDR']  # For Colab
    # tpu='your-tpu-name'  # For Cloud TPU
)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)
print(f'Number of TPU cores: {strategy.num_replicas_in_sync}')

# Create model inside scope
with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        steps_per_execution=100  # Important for TPU performance
    )

# TPU requires datasets (not numpy arrays)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(50000).batch(
    128 * strategy.num_replicas_in_sync,
    drop_remainder=True
).prefetch(tf.data.AUTOTUNE)

model.fit(train_dataset, epochs=10, steps_per_epoch=500)

# Important TPU considerations:
# 1. TPU requires static shapes - use drop_remainder=True in batch()
# 2. Use steps_per_execution > 1 in compile() for performance
# 3. All data must be tf.data.Dataset (no numpy arrays)
# 4. Functions passed to strategy.run() must be @tf.function compatible
```

### TPU Initialization

```python
# Full TPU initialization sequence
import tensorflow as tf

# Step 1: Create resolver
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

# Step 2: Connect to cluster
tf.config.experimental_connect_to_cluster(resolver)

# Step 3: Initialize TPU system
tf.tpu.experimental.initialize_tpu_system(resolver)

# Step 4: Create strategy
strategy = tf.distribute.TPUStrategy(resolver)

# Step 5: Verify
print(f'All TPU devices: {tf.config.list_logical_devices("TPU")}')
print(f'Number of replicas: {strategy.num_replicas_in_sync}')
```

---

## ParameterServerStrategy

Asynchronous training with parameter servers. Workers compute gradients and parameter servers store and update variables.

```python
tf.distribute.ParameterServerStrategy(
    cluster_resolver,                  # ClusterResolver (required)
    variable_partitioner=None          # Partitioner for large variables
)
```

### Architecture

- **Coordinator**: Creates resources, dispatches functions, handles checkpoints
- **Workers**: Execute training functions, read/write variables from/to parameter servers
- **Parameter Servers**: Store variables, apply gradient updates

### Usage with Custom Training Loop

```python
# Set up cluster
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

# Create strategy
strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=256 << 10,     # 256 KB minimum per shard
        max_shards=10                  # Max number of shards per variable
    )
)

# Create coordinator
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)

# Create model inside scope
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define per-worker dataset
def dataset_fn(input_context):
    global_batch_size = 64 * strategy.num_replicas_in_sync
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)

# Define training step
@tf.function
def step_fn(iterator):
    def replica_fn(batch_data):
        x, y = batch_data
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, predictions, from_logits=True
            )
            loss = tf.nn.compute_average_loss(
                loss, global_batch_size=global_batch_size
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    per_replica_loss = strategy.run(replica_fn, args=(next(iterator),))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

# Create per-worker iterator
per_worker_dataset = coordinator.create_per_worker_dataset(dist_dataset)
per_worker_iterator = iter(per_worker_dataset)

# Train
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
    coordinator.join()
    print(f'Epoch {epoch}: waiting for tasks to complete')
```

### Usage with Keras Model.fit

```python
# Create dataset creator
def dataset_fn(input_context):
    global_batch_size = 64
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(10000).batch(batch_size)
    return dataset

dataset_creator = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

# Create and compile model
with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        steps_per_execution=10
    )

# Train
model.fit(
    dataset_creator,
    epochs=10,
    steps_per_epoch=1000
)
```

### Variable Partitioning

```python
# Partition large variables across parameter servers
partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
    min_shard_bytes=256 << 10,     # 256KB minimum shard size
    max_shards=10
)

strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=partitioner
)

# Available partitioners:
# - MinSizePartitioner: Partitions to minimum shard size
# - MaxSizePartitioner: Limits maximum shard size
# - FixedShardsPartitioner: Fixed number of shards
```

---

## CentralStorageStrategy

Single-machine strategy that places variables on CPU or a single GPU, with computation replicated across all devices.

```python
tf.distribute.experimental.CentralStorageStrategy(
    compute_devices=None,              # List of device strings for computation
    parameter_device=None              # Device string for variable storage
)
```

### Usage

```python
# Use all GPUs for computation, CPU for variable storage
strategy = tf.distribute.experimental.CentralStorageStrategy()

# Specify devices
strategy = tf.distribute.experimental.CentralStorageStrategy(
    compute_devices=['/gpu:0', '/gpu:1'],
    parameter_device='/cpu:0'
)

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, epochs=10)
```

**When to use:** When the model is too large to fit entirely on each GPU, or when you want to avoid the memory overhead of variable replication.

---

## OneDeviceStrategy

A strategy for testing and debugging that uses a single device.

```python
tf.distribute.OneDeviceStrategy(
    device='/gpu:0'                    # Device string
)
```

### Usage

```python
# Test on a single GPU
strategy = tf.distribute.OneDeviceStrategy('/gpu:0')

# Test on CPU
strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

# Use to verify distributed code works
with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, epochs=2)
```

**When to use:** For testing, debugging distributed code on a single device, or ensuring code works with the strategy API without needing multiple devices.

---

## ClusterResolver

Cluster resolvers discover cluster configuration from various environments.

### TFConfigClusterResolver

Reads cluster configuration from the `TF_CONFIG` environment variable.

```python
tf.distribute.cluster_resolver.TFConfigClusterResolver(
    task_type=None,                    # Override task type
    task_id=None,                      # Override task id
    rpc_layer='grpc',                  # RPC layer ('grpc' or 'grpc+verbs')
    environment=None
)
```

**TF_CONFIG format:**
```json
{
    "cluster": {
        "worker": ["host1:port1", "host2:port2"],
        "ps": ["host3:port3"]
    },
    "task": {
        "type": "worker",
        "index": 0
    }
}
```

### GCEClusterResolver

Discovers cluster from GCE (Google Compute Engine) metadata.

```python
tf.distribute.cluster_resolver.GCEClusterResolver(
    task_type='worker',
    task_id=0,
    rpc_layer='grpc',
    gce_instance=None
)
```

### KubernetesClusterResolver

Discovers cluster from Kubernetes service endpoints.

```python
tf.distribute.cluster_resolver.KubernetesClusterResolver(
    job_to_label_mapping=None,
    task_type='worker',
    task_id=0,
    rpc_layer='grpc',
    namespace='default'
)
```

### TPUClusterResolver

Discovers TPU cluster configuration.

```python
tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=None,                          # TPU name or address. None = auto-detect
    zone=None,                         # GCP zone
    project=None,                      # GCP project
    job_name='worker',
    coordinator_address=None,
    credentials=None,
    service=None,
    discovery_url=None
)
```

### SlurmClusterResolver

Discovers cluster from Slurm environment variables.

```python
tf.distribute.cluster_resolver.SlurmClusterResolver(
    jobs=None,                         # Dict mapping job names to tasks
    port_base=8888,
    gpus_per_node=1,
    gpus_per_task=1,
    task_type='worker',
    task_id=0,
    rpc_layer='grpc',
    auto_set_gpu=True
)
```

### ClusterResolver Common Methods

```python
# Get cluster spec
cluster_spec = resolver.cluster_spec()
# Returns ClusterSpec with cluster configuration

# Get task info
task_type = resolver.task_type     # 'worker', 'ps', 'chief'
task_id = resolver.task_id         # Integer task index

# Get master address
master = resolver.master(
    task_type=None,
    task_id=None,
    rpc_layer=None
)

# Number of accelerators
num_accelerators = resolver.num_accelerators()
# Returns dict {'GPU': 4} or {'TPU': 8}
```

---

## Cross-Device Operations

### NcclAllReduce

Uses NVIDIA NCCL library for efficient GPU all-reduce.

```python
tf.distribute.NcclAllReduce(
    num_packs=2,                       # Number of gradient packs
    shared_name=None
)
```

### CollectiveAllReduce

Uses TensorFlow's collective operations for all-reduce. Works on CPU and GPU.

```python
tf.distribute.CollectiveAllReduce(
    group_size=None,
    group_key=None
)
```

### HierarchicalCopyAllReduce

Hierarchical all-reduce. Efficient for many GPUs.

```python
tf.distribute.HierarchicalCopyAllReduce(
    num_packs=3                        # Number of gradient packs
)
```

### ReduceToOneDevice

Reduces to a single device. Simple but not scalable.

```python
tf.distribute.ReduceToOneDevice(
    reduce_device='/gpu:0',
    accumulation_fn=tf.math.add
)
```

### MirroredStrategyDeviceState

Manages device state for mirrored variables.

```python
tf.distribute.MirroredStrategyDeviceState(
    device_map=None,
    logical_device=None
)
```

---

## Distributed Variables

### AggregatingVariable

A variable that aggregates values across replicas. Created automatically when using `MirroredStrategy`.

```python
# Created automatically inside strategy.scope()
with strategy.scope():
    v = tf.Variable(0.0)
    # v is an AggregatingVariable on MirroredStrategy
```

### SyncOnReadVariable

A variable that synchronizes only when read in cross-replica context. Used for batch normalization statistics, metrics, etc.

```python
# Created with synchronization=ON_READ
with strategy.scope():
    v = tf.Variable(
        0.0,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.MEAN
    )
```

### Aggregation Types

```python
tf.VariableAggregation
# Values:
#   NONE             - No aggregation
#   SUM              - Sum across replicas
#   MEAN             - Mean across replicas
#   ONLY_FIRST_REPLICA - Use value from first replica only
```

### Synchronization Types

```python
tf.VariableSynchronization
# Values:
#   ON_WRITE  - Synchronize on every write (default, for mirrored variables)
#   ON_READ   - Synchronize on every read (for metrics, batch norm stats)
#   NONE      - No synchronization
```

### Working with Distributed Variables

```python
with strategy.scope():
    # Mirrored variable (replicated on all devices)
    mirrored_var = tf.Variable(
        1.0,
        synchronization=tf.VariableSynchronization.ON_WRITE,
        aggregation=tf.VariableAggregation.MEAN
    )

    # SyncOnRead variable (for metrics)
    metric_var = tf.Variable(
        0.0,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.SUM
    )

    # Sharded variable (for parameter server)
    # Large variables can be sharded across parameter servers
    embedding = tf.keras.layers.Embedding(100000, 128)
```

---

## Input Distribution

### strategy.experimental_distribute_dataset

Wraps a tf.data.Dataset for distributed training.

```python
dist_dataset = strategy.experimental_distribute_dataset(
    dataset,                           # tf.data.Dataset with GLOBAL batch size
    options=None                       # tf.distribute.InputOptions
)
```

**Usage:**
```python
strategy = tf.distribute.MirroredStrategy()

# Create dataset with global batch size
global_batch_size = 32 * strategy.num_replicas_in_sync
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(10000).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

# Distribute
dist_dataset = strategy.experimental_distribute_dataset(dataset)

# Iterate
for batch in dist_dataset:
    strategy.run(train_step, args=(batch,))
```

### strategy.distribute_datasets_from_function

Creates distributed datasets from a function, allowing per-replica customization.

```python
dist_dataset = strategy.distribute_datasets_from_function(
    dataset_fn,                        # Function taking InputContext, returns tf.data.Dataset
    options=None                       # tf.distribute.InputOptions
)
```

**Usage:**
```python
def dataset_fn(input_context):
    # input_context provides:
    #   input_context.num_input_pipelines - total number of workers
    #   input_context.input_pipeline_id - this worker's ID
    #   input_context.num_replicas_in_sync - total replicas
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # Shard data across workers
    dataset = dataset.shard(
        input_context.num_input_pipelines,
        input_context.input_pipeline_id
    )
    dataset = dataset.shuffle(10000).batch(batch_size)
    return dataset

dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
```

### InputOptions

```python
tf.distribute.InputOptions(
    experimental_replication_mode=tf.distribute.InputReplicationMode.PER_WORKER,
    # PER_WORKER  - Each worker gets its own data (default)
    # PER_REPLICA - Each replica gets its own data (only with distribute_datasets_from_function)

    experimental_place_dataset_on_device=False,  # Place dataset on worker device
    experimental_per_replica_buffer_size=1,       # Buffer size per replica
    experimental_slack=False                       # Use slack for overlapping
)
```

---

## Custom Training Loops (Distributed)

### Basic Distributed Training Loop

```python
strategy = tf.distribute.MirroredStrategy()
print(f'Number of GPUs: {strategy.num_replicas_in_sync}')

# Global batch size
global_batch_size = 64 * strategy.num_replicas_in_sync

# Create model and optimizer in scope
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE  # Must be NONE for distributed
    )
    train_loss_metric = tf.keras.metrics.Mean()
    train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Define dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(50000).batch(global_batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Define loss computation
def compute_loss(labels, predictions):
    per_example_loss = loss_fn(labels, predictions)
    # Compute average loss across replicas
    return tf.nn.compute_average_loss(
        per_example_loss,
        global_batch_size=global_batch_size
    )

# Training step
@tf.function
def train_step(batch):
    x, y = batch

    def step_fn(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = compute_loss(y, predictions)
            # Add regularization losses
            loss += tf.nn.scale_regularization_loss(sum(model.losses))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, predictions

    per_replica_loss, per_replica_preds = strategy.run(step_fn, args=(x, y))

    # Reduce metrics across replicas
    total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    # Update metrics
    train_loss_metric.update_state(total_loss)
    train_accuracy_metric.update_state(y, per_replica_preds)

    return total_loss

# Training loop
for epoch in range(num_epochs):
    train_loss_metric.reset_state()
    train_accuracy_metric.reset_state()

    for batch in dist_dataset:
        loss = train_step(batch)

    print(f'Epoch {epoch}: Loss={train_loss_metric.result():.4f}, '
          f'Accuracy={train_accuracy_metric.result():.4f}')
```

### Key Functions for Distributed Loss

```python
# compute_average_loss - Averages per-example loss across replicas
tf.nn.compute_average_loss(
    per_example_loss,                  # Loss tensor from each replica
    global_batch_size=None,            # Required
    num_replicas_in_sync=None          # Optional, auto-detected
)

# scale_regularization_loss - Scales regularization loss for distributed training
tf.nn.scale_regularization_loss(
    regularization_loss                # Regularization loss tensor
)

# nchw_to_nhwc / nhwc_to_nchw
# For converting between data formats across devices
```

### Per-Replica Loss Handling

```python
# CORRECT: Compute per-example loss, then average
def compute_loss(labels, predictions, model_losses):
    per_example_loss = loss_fn(labels, predictions)
    loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
    if model_losses:
        loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    return loss

# WRONG: Do NOT reduce loss before compute_average_loss
# loss = tf.reduce_mean(per_example_loss)  # DON'T DO THIS

# For custom loss with sample weights
def weighted_loss(labels, predictions, sample_weight):
    per_example_loss = loss_fn(labels, predictions)
    if sample_weight is not None:
        per_example_loss *= sample_weight
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
```

---

## Keras Integration

### Strategy with Keras Model.fit

```python
# Step 1: Create strategy
strategy = tf.distribute.MirroredStrategy()

# Step 2: Create model inside scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        steps_per_execution=10  # Optional: improves performance
    )

# Step 3: Create dataset with global batch size
global_batch_size = 32 * strategy.num_replicas_in_sync
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(50000).batch(global_batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# Step 4: Train
model.fit(train_dataset, epochs=10)
```

### Saving and Loading with Strategy

```python
# Save inside strategy scope
with strategy.scope():
    model.save('model.keras')

# Load for inference (no strategy needed)
model = tf.keras.models.load_model('model.keras')

# Load for continued distributed training
with strategy.scope():
    model = tf.keras.models.load_model('model.keras')
```

### Multi-Worker Keras Training

```python
# Each worker runs this same code
import os
import json
import tensorflow as tf

# TF_CONFIG is set differently for each worker
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Determine number of steps per epoch
num_workers = strategy.num_replicas_in_sync
global_batch_size = 64 * num_workers

# Callbacks - only chief should save
callbacks = []
if 'chief' in os.environ.get('TF_CONFIG', ''):
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('model.keras'))

train_dataset = create_dataset(global_batch_size)
model.fit(train_dataset, epochs=10, callbacks=callbacks)
```

---

## Multi-Worker Setup

### TF_CONFIG Configuration

The `TF_CONFIG` environment variable tells each worker about the cluster.

```json
// Chief worker (worker 0)
{
    "cluster": {
        "worker": ["worker0.example.com:12345", "worker1.example.com:12345"]
    },
    "task": {
        "type": "worker",
        "index": 0
    }
}

// Worker 1
{
    "cluster": {
        "worker": ["worker0.example.com:12345", "worker1.example.com:12345"]
    },
    "task": {
        "type": "worker",
        "index": 1
    }
}

// Parameter server (for ParameterServerStrategy)
{
    "cluster": {
        "worker": ["worker0:12345", "worker1:12345"],
        "ps": ["ps0:12345"]
    },
    "task": {
        "type": "ps",
        "index": 0
    }
}
```

### Setting TF_CONFIG in Python

```python
import os
import json

# For worker 0
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['localhost:12345', 'localhost:12346']
    },
    'task': {'type': 'worker', 'index': 0}
})

# For worker 1
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['localhost:12345', 'localhost:12346']
    },
    'task': {'type': 'worker', 'index': 1}
})
```

### Cluster Spec Utilities

```python
# Create ClusterSpec from dictionary
cluster_spec = tf.train.ClusterSpec({
    'worker': ['host1:port1', 'host2:port2'],
    'ps': ['host3:port3']
})

# Normalize cluster spec
cluster_spec = tf.distribute.multi_worker_util.normalize_cluster_spec(cluster_dict)

# Check if this is chief
resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
is_chief = (resolver.task_type == 'worker' and resolver.task_id == 0)
# OR for ParameterServerStrategy:
is_chief = (resolver.task_type == 'chief')
```

### Task Types

```python
# For MultiWorkerMirroredStrategy:
# - 'worker': All workers are equal
# - Worker with index 0 acts as chief

# For ParameterServerStrategy:
# - 'chief': The chief worker (coordinator)
# - 'worker': Training workers
# - 'ps': Parameter servers
```

---

## Performance Tuning

### Gradient Aggregation

```python
# Control gradient aggregation with cross_device_ops
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.NcclAllReduce(num_packs=2)
)

# More packs = smaller all-reduce payloads = better overlap with compute
# Typical values: 1-8, default is 2
```

### Overlapping Computation and Communication

```python
# Enable experimental gradient packing
options = tf.data.Options()
options.experimental_optimization.parallel_batch = True
dataset = dataset.with_options(options)

# Use CommunicationOptions for multi-worker
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL,
        bytes_per_pack=0,
        max_parallel_allreduce=4,
        max_parallel_broadcasts=4
    )
)
```

### Batch Size Guidelines

```python
# Global batch size should be per_gpu_batch * num_gpus
per_gpu_batch = 32
num_gpus = strategy.num_replicas_in_sync
global_batch_size = per_gpu_batch * num_gpus

# For multi-worker:
# global_batch_size = per_gpu_batch * gpus_per_worker * num_workers

# Tips:
# 1. Use drop_remainder=True for TPU and for consistent shapes
# 2. Scale learning rate linearly with batch size (linear scaling rule)
# 3. Use warmup for large batch training
```

### Mixed Precision with Distributed Training

```python
# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Loss scaling for numerical stability
with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(0.001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
```

### Profiling Distributed Training

```python
# Enable profiling
tf.profiler.experimental.start('logdir')

# Profile specific steps
for epoch in range(epochs):
    for step, batch in enumerate(dist_dataset):
        if step == 5:
            tf.profiler.experimental.start('logdir')
        train_step(batch)
        if step == 15:
            tf.profiler.experimental.stop()

tf.profiler.experimental.stop()

# View with TensorBoard
# tensorboard --logdir=logdir
```

---

## Fault Tolerance

### Checkpointing for Fault Tolerance

```python
# Save checkpoints regularly
checkpoint_dir = '/tmp/training_checkpoints'
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Restore from latest checkpoint
if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f'Restored from {manager.latest_checkpoint}')

# Save during training
for epoch in range(num_epochs):
    for step, batch in enumerate(dist_dataset):
        train_step(batch)
        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 1000 == 0:
            manager.save()
```

### BackupAndRestore Callback

```python
# Automatic backup and restore for Keras training
callback = tf.keras.callbacks.BackupAndRestore(
    backup_dir='/tmp/backup',
    save_freq='epoch',                 # 'epoch' or integer (batch frequency)
    delete_checkpoint=True,            # Delete backup after successful training
    save_before_preemption=False       # Save on preemption signal (experimental)
)

model.fit(
    train_dataset,
    epochs=100,
    callbacks=[callback]
)
```

### Preemption Handling

```python
# Handle preemption in custom training loop
import signal

class PreemptionHandler:
    def __init__(self, checkpoint_manager):
        self.manager = checkpoint_manager
        signal.signal(signal.SIGTERM, self._handle_preemption)
        signal.signal(signal.SIGINT, self._handle_preemption)

    def _handle_preemption(self, signum, frame):
        print(f'Received signal {signum}, saving checkpoint...')
        self.manager.save()
        exit(0)

# Usage
handler = PreemptionHandler(manager)
```

### Multi-Worker Fault Tolerance

```python
# For MultiWorkerMirroredStrategy
# If a worker fails, the entire cluster needs to restart from checkpoint

# Key considerations:
# 1. Save checkpoints frequently
# 2. Use BackupAndRestore callback
# 3. Design training to be idempotent (same result regardless of restart point)

# Example fault-tolerant setup
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Restore
initial_epoch = 0
if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    initial_epoch = int(checkpoint.epoch)

# Training with resume capability
for epoch in range(initial_epoch, num_epochs):
    for batch in dist_dataset:
        train_step(batch)

    checkpoint.epoch.assign(epoch)
    manager.save()
```

---

## DTensor

DTensor provides a way to distribute tensors and computations across multiple devices with a mesh-based approach. It supports both data parallelism and model parallelism (SPMD - Single Program, Multiple Data).

### DTensor Concepts

```python
import tensorflow as tf
from tensorflow.experimental import dtensor
```

### Mesh

A Mesh describes how a set of devices are arranged into a logical grid.

```python
# Create a mesh
# 1D mesh (data parallelism)
mesh_1d = dtensor.create_mesh(
    mesh_name='1d_mesh',
    devices=['CPU:0', 'CPU:1', 'CPU:2', 'CPU:3'],
    mesh_dims=[('batch', 4)]
)
# Devices arranged as: [0, 1, 2, 3] along 'batch' dimension

# 2D mesh (data + model parallelism)
mesh_2d = dtensor.create_mesh(
    mesh_name='2d_mesh',
    devices=['CPU:0', 'CPU:1', 'CPU:2', 'CPU:3'],
    mesh_dims=[('batch', 2), ('model', 2)]
)
# Devices arranged as:
#   batch=0: [CPU:0, CPU:1]
#   batch=1: [CPU:2, CPU:3]
#   model=0: [CPU:0, CPU:2]
#   model=1: [CPU:1, CPU:3]
```

### Layout

A Layout describes how a tensor is distributed across a mesh.

```python
# Sharded layout - tensor is split along a dimension
layout_sharded = dtensor.Layout(
    ['sharded/batch', 'unsharded'],    # First dim sharded along 'batch', second dim unsharded
    mesh=mesh_1d
)

# Replicated layout - same tensor on all devices
layout_replicated = dtensor.Layout(
    ['unsharded', 'unsharded'],
    mesh=mesh_1d
)

# Fully sharded layout
layout_fully_sharded = dtensor.Layout(
    ['sharded/batch', 'sharded/batch'],
    mesh=mesh_1d
)
```

### DTensor Operations

```python
# Create a DTensor
a = dtensor.call_with_layout(
    tf.ones,
    layout=dtensor.Layout(['sharded/batch'], mesh_1d),
    shape=(8,),
    dtype=tf.float32
)

# DTensor from numpy
data = np.random.random((8, 4))
layout = dtensor.Layout(['sharded/batch', 'unsharded'], mesh_1d)
dt = dtensor.from_numpy(data, layout=layout)

# Regular TensorFlow ops work with DTensors
result = tf.matmul(a, b)  # Automatically distributes the computation

# Relayout (change how tensor is distributed)
new_layout = dtensor.Layout(['unsharded', 'sharded/batch'], mesh_1d)
relaid = dtensor.relayout(dt, new_layout)
```

### SPMD with DTensor

```python
# Single Program, Multiple Data approach
# Write the same code; DTensor handles distribution

mesh = dtensor.create_mesh(
    devices=tf.config.list_physical_devices('GPU'),
    mesh_dims=[('batch', 4)]
)

# Create model with DTensor layout
class DTensorDense(tf.keras.layers.Layer):
    def __init__(self, units, layout=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.layout = layout

    def build(self, input_shape):
        self.kernel = dtensor.DVariable(
            dtensor.call_with_layout(
                tf.random.normal,
                self.layout or dtensor.Layout(['unsharded', 'unsharded'], mesh),
                shape=(input_shape[-1], self.units)
            )
        )
        self.bias = dtensor.DVariable(
            dtensor.call_with_layout(
                tf.zeros,
                self.layout or dtensor.Layout(['unsharded'], mesh),
                shape=(self.units,)
            )
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias
```

### DTensor Keras Integration

```python
# DTensor-enabled model training
with dtensor.run_on(mesh):
    model = tf.keras.Sequential([
        dtensor.DTensorDense(128, activation='relu',
                            layout=dtensor.Layout(['sharded/batch', 'unsharded'], mesh)),
        dtensor.DTensorDense(10, activation='softmax',
                            layout=dtensor.Layout(['sharded/batch', 'unsharded'], mesh))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(train_dataset, epochs=10)
```

### DTensor Mesh Visualization

```python
# Check mesh configuration
print(mesh.device_ids())      # List of device IDs
print(mesh.shape())           # Mesh dimensions
print(mesh.size())            # Total number of devices

# Check layout
print(layout.rank)            # Number of dimensions
print(layout.sharding_specs)  # Sharding specification per dimension
print(layout.mesh)            # Associated mesh
```

---

## Strategy Selection Guide

| Strategy | Machines | Synchronization | Use Case |
|----------|----------|----------------|----------|
| `MirroredStrategy` | Single machine | Synchronous | Multi-GPU training |
| `MultiWorkerMirroredStrategy` | Multiple machines | Synchronous | Multi-node GPU training |
| `TPUStrategy` | TPU Pod | Synchronous | TPU training |
| `ParameterServerStrategy` | Multiple machines | Asynchronous | Large-scale async training |
| `CentralStorageStrategy` | Single machine | Synchronous | Large models, limited GPU memory |
| `OneDeviceStrategy` | Single device | N/A | Testing, debugging |

### Common Patterns by Scale

```python
# Single GPU
strategy = tf.distribute.OneDeviceStrategy('/gpu:0')

# Single machine, multiple GPUs
strategy = tf.distribute.MirroredStrategy()

# Multiple machines, multiple GPUs each
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# TPU
strategy = tf.distribute.TPUStrategy(resolver)

# Parameter servers (large scale, async)
strategy = tf.distribute.ParameterServerStrategy(resolver)
```
