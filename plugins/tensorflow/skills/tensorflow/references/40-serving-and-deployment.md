# TensorFlow Serving and Deployment Reference

## Table of Contents

1. [TensorFlow Serving Architecture](#tensorflow-serving-architecture)
2. [REST API](#rest-api)
3. [gRPC API](#grpc-api)
4. [Batch Scheduling](#batch-scheduling)
5. [Model Versioning](#model-versioning)
6. [Docker Deployment](#docker-deployment)
7. [TensorFlow Hub](#tensorflow-hub)
8. [TensorFlow.js](#tensorflowjs)
9. [Mobile Deployment](#mobile-deployment)
10. [Cloud Deployment](#cloud-deployment)
11. [Kubernetes Deployment](#kubernetes-deployment)
12. [Edge Deployment](#edge-deployment)
13. [Model Optimization Toolkit](#model-optimization-toolkit)
14. [Export Patterns](#export-patterns)

---

## TensorFlow Serving Architecture

### Overview

TensorFlow Serving is a flexible, high-performance serving system for machine
learning models, designed for production environments. It provides:

- **Out-of-the-box production readiness**: Handles concurrent requests,
  batching, and model lifecycle management.
- **Multiple model serving**: Serve multiple models or multiple versions
  simultaneously.
- **Model versioning**: Hot-swap model versions without downtime.
- **gRPC and REST APIs**: Multiple protocol support.
- **Batching**: Automatic request batching for GPU efficiency.
- **Extensible architecture**: Plugin system for custom servables.

### Core Concepts

**Servable**: The fundamental serving unit. Typically a TensorFlow model
(loaded from a SavedModel), but can be any type of inference artifact.

**Servable Version**: A specific version of a servable. TensorFlow Serving
manages the lifecycle of multiple versions concurrently.

**Servable Stream**: A sequence of versions of the same servable, ordered by
version number.

**Manager**: Manages the full lifecycle of servables: loading, serving,
  unloading.

**Loader**: Handles loading and unloading of a specific servable version.

**Source**: Plugin that finds and creates loaders for servable versions.

**Aspired Versions**: Set of versions that should be loaded, as determined by
  a Source.

### Architecture Diagram

```
Client Request
     |
     v
+------------------+
|   REST/gRPC API  |
+------------------+
     |
     v
+------------------+
|  Serving Core    |
|  - Manager       |
|  - Loader        |
|  - Version       |
|    Policy        |
+------------------+
     |
     v
+------------------+
|  Model Storage   |
|  - File system   |
|  - Cloud storage |
+------------------+
```

### ModelServer Configuration

```protobuf
# models.config
model_config_list {
  config {
    name: "my_model"
    base_path: "/models/my_model"
    model_platform: "tensorflow"
    model_version_policy {
      specific {
        versions: 1
        versions: 2
      }
    }
    version_labels {
      key: "stable"
      value: 2
    }
    version_labels {
      key: "canary"
      value: 3
    }
  }
}
```

### Starting ModelServer

```bash
# Basic server
tensorflow_model_server --model_base_path=/models/my_model

# With configuration file
tensorflow_model_server --model_config_file=/path/to/models.config

# With REST API on port 8501
tensorflow_model_server \
    --rest_api_port=8501 \
    --model_config_file=/path/to/models.config

# With both gRPC and REST
tensorflow_model_server \
    --grpc_port=8500 \
    --rest_api_port=8501 \
    --model_config_file=/path/to/models.config

# Enable batching
tensorflow_model_server \
    --enable_batching=true \
    --batching_parameters_file=/path/to/batching.config
```

---

## REST API

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/models/{model}[:predict\|:classify\|:regress]` | POST | Model inference |
| `/v1/models/{model}/versions/{version}[:action]` | POST | Versioned inference |
| `/v1/models/{model}/metadata` | GET | Model metadata |
| `/v1/models/{model}/versions/{version}/metadata` | GET | Versioned metadata |
| `/v1/models/{model}` | GET | Model status |

### Predict API

```bash
# Predict request
curl -X POST http://localhost:8501/v1/models/my_model:predict \
    -H "Content-Type: application/json" \
    -d '{
        "instances": [
            {"input": [1.0, 2.0, 3.0]},
            {"input": [4.0, 5.0, 6.0]}
        ]
    }'
```

**Request format**:

```json
{
    "instances": [
        {"input_key_1": [value], "input_key_2": [value]},
        {"input_key_1": [value], "input_key_2": [value]}
    ]
}
```

For models with a single input:

```json
{
    "instances": [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
}
```

**Response format**:

```json
{
    "predictions": [
        {"output_key": [0.1, 0.9]},
        {"output_key": [0.8, 0.2]}
    ]
}
```

### Classify API

```bash
curl -X POST http://localhost:8501/v1/models/my_model:classify \
    -H "Content-Type: application/json" \
    -d '{
        "examples": [
            {"age": 25, "income": 50000},
            {"age": 35, "income": 75000}
        ]
    }'
```

**Response**:

```json
{
    "results": [
        [{"label": "class_a", "score": 0.9}, {"label": "class_b", "score": 0.1}],
        [{"label": "class_a", "score": 0.3}, {"label": "class_b", "score": 0.7}]
    ]
}
```

### Regress API

```bash
curl -X POST http://localhost:8501/v1/models/my_model:regress \
    -H "Content-Type: application/json" \
    -d '{
        "examples": [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0}
        ]
    }'
```

**Response**:

```json
{
    "results": [2.5, 7.0]
}
```

### Model Status

```bash
# Get model status
curl http://localhost:8501/v1/models/my_model

# Response
{
    "model_version_status": [
        {
            "version": "2",
            "state": "READY",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        }
    ]
}
```

### Model Metadata

```bash
curl http://localhost:8501/v1/models/my_model/metadata

# Response
{
    "model_spec": {
        "name": "my_model",
        "signature_name": "",
        "version": "2"
    },
    "metadata": {
        "signature_def": {
            "serving_default": {
                "inputs": {
                    "input": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "784"}]}}
                },
                "outputs": {
                    "output": {"dtype": "DT_FLOAT", "tensor_shape": {"dim": [{"size": "-1"}, {"size": "10"}]}}
                },
                "method_name": "tensorflow/serving/predict"
            }
        }
    }
}
```

---

## gRPC API

### PredictionService

Defined in `tensorflow_serving/apis/prediction_service.proto`:

```protobuf
service PredictionService {
    // Predict
    rpc Predict(PredictRequest) returns (PredictResponse);

    // Classify
    rpc Classify(ClassificationRequest) returns (ClassificationResponse);

    // Regress
    rpc Regress(RegressionRequest) returns (RegressionResponse);

    // Multi-inference
    rpc MultiInference(MultiInferenceRequest) returns (MultiInferenceResponse);

    // Get model status
    rpc GetModelStatus(GetModelStatusRequest) returns (GetModelStatusResponse);

    // Get model metadata
    rpc GetModelMetadata(GetModelMetadataRequest) returns (GetModelMetadataResponse);
}
```

### Predict Request/Response

```protobuf
message PredictRequest {
    ModelSpec model_spec = 1;
    map<string, TensorProto> inputs = 2;
    // Optional: filter output names
    google.protobuf.FieldMask output_filter = 3;
}

message PredictResponse {
    map<string, TensorProto> outputs = 1;
    ModelSpec model_spec = 2;
}
```

### gRPC Client Example (Python)

```python
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

# Create gRPC channel
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create predict request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'

# Set input tensor
input_data = np.random.randn(1, 784).astype(np.float32)
request.inputs['input'].CopyFrom(
    tf.make_tensor_proto(input_data, dtype=tf.float32))

# Call predict
response = stub.Predict(request, timeout=10.0)

# Parse output
output = tf.make_ndarray(response.outputs['output'])
print(output.shape)  # (1, 10)
```

### Get Model Status (gRPC)

```python
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import model_service_pb2_grpc

status_stub = model_service_pb2_grpc.ModelServiceStub(channel)

status_request = get_model_status_pb2.GetModelStatusRequest()
status_request.model_spec.name = 'my_model'

status_response = status_stub.GetModelStatus(status_request)
for version_status in status_response.model_version_status:
    print(f"Version: {version_status.version}, State: {version_status.state}")
```

### Get Model Metadata (gRPC)

```python
from tensorflow_serving.apis import get_model_metadata_pb2

metadata_request = get_model_metadata_pb2.GetModelMetadataRequest()
metadata_request.model_spec.name = 'my_model'
metadata_request.metadata_field.append('signature_def')

metadata_response = stub.GetModelMetadata(metadata_request)
# Parse signature def from metadata_response
```

---

## Batch Scheduling

### Batching Configuration

```protobuf
# batching.config
max_batch_size { value: 128 }
batch_timeout_micros { value: 10000 }
max_enqueued_batches { value: 100 }
num_batch_threads { value: 4 }
pad_variable_length_inputs: true
}
```

### Batching Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_batch_size` | 128 | Maximum batch size |
| `batch_timeout_micros` | 0 | Max wait time before processing a partial batch |
| `max_enqueued_batches` | 100 | Maximum queued batches |
| `num_batch_threads` | 1 | Number of batch processing threads |
| `pad_variable_length_inputs` | false | Pad inputs to same size in batch |

### Session Batching

For models that benefit from batching on GPU:

```bash
tensorflow_model_server \
    --enable_batching=true \
    --batching_parameters_file=batching.config
```

### Batching in Python Client

```python
# Use batched inference for throughput
import concurrent.futures

def predict_batch(inputs):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'my_model'
    request.inputs['input'].CopyFrom(
        tf.make_tensor_proto(inputs, dtype=tf.float32))
    response = stub.Predict(request, timeout=10.0)
    return tf.make_ndarray(response.outputs['output'])

# Send requests in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(predict_batch, batch) for batch in data_batches]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

---

## Model Versioning

### Version Policy

```protobuf
// Latest version policy
model_version_policy {
    latest {
        num_versions: 2  // Keep 2 latest versions
    }
}

// Specific versions
model_version_policy {
    specific {
        versions: 1
        versions: 2
        versions: 5
    }
}

// All versions
model_version_policy {
    all {}
}
```

### Version Labels

```protobuf
config {
    name: "my_model"
    base_path: "/models/my_model"
    model_platform: "tensorflow"
    version_labels {
        key: "stable"
        value: 2
    }
    version_labels {
        key: "canary"
        value: 3
    }
}
```

Use labels in requests:

```bash
# Use stable version
curl http://localhost:8501/v1/models/my_model/labels/stable:predict

# Use canary version
curl http://localhost:8501/v1/models/my_model/labels/canary:predict
```

### Canary Deployment

```bash
# Directory structure
/models/my_model/
    1/    # Current production version
        saved_model.pb
        variables/
    2/    # New canary version
        saved_model.pb
        variables/

# TF Serving automatically detects new versions
# and can serve both simultaneously
```

### Rollback

```bash
# Delete the problematic version directory
rm -rf /models/my_model/2/

# TF Serving automatically falls back to version 1
# Or explicitly configure version policy
```

---

## Docker Deployment

### Official Docker Images

```bash
# CPU only
docker pull tensorflow/serving:latest

# GPU support
docker pull tensorflow/serving:latest-gpu
```

### Basic Docker Deployment

```bash
# Serve a model
docker run -d --name tf-serving \
    -p 8500:8500 \
    -p 8501:8501 \
    -v /path/to/models:/models/my_model \
    -e MODEL_NAME=my_model \
    tensorflow/serving:latest

# The model directory should contain versioned subdirectories:
# /models/my_model/1/saved_model.pb
```

### Docker with Configuration

```bash
docker run -d --name tf-serving \
    -p 8500:8500 \
    -p 8501:8501 \
    -v /path/to/models.config:/models/models.config \
    -v /path/to/models:/models \
    tensorflow/serving:latest \
    --model_config_file=/models/models.config
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3'
services:
  tf-serving:
    image: tensorflow/serving:latest-gpu
    ports:
      - "8500:8500"
      - "8501:8501"
    volumes:
      - ./models:/models
      - ./config:/config
    environment:
      - MODEL_NAME=my_model
    command: >
      --model_config_file=/config/models.config
      --enable_batching=true
      --batching_parameters_file=/config/batching.config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### Building Custom Images

```dockerfile
FROM tensorflow/serving:latest

# Copy model
COPY ./exported_model /models/my_model/1

# Set environment
ENV MODEL_NAME=my_model

# Expose ports
EXPOSE 8500 8501

# Default entrypoint from base image
```

---

## TensorFlow Hub

### Overview

TensorFlow Hub (tfhub.dev) is a repository of reusable machine learning
modules. Modules are self-contained pieces of TensorFlow graph with
pre-trained weights.

### Loading Modules

```python
import tensorflow_hub as hub

# Load a module
module = hub.load("https://tfhub.dev/google/nnlm-en-dim50/2")

# Use as a Keras layer
embedding_layer = hub.KerasLayer(
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    input_shape=[],  # Text input
    dtype=tf.string,
    trainable=False)

# Build model with Hub layer
model = tf.keras.Sequential([
    embedding_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Module Signatures

```python
# List available signatures
module = hub.load("https://tfhub.dev/google/nnlm-en-dim50/2")
print(list(module.signatures.keys()))

# Use specific signature
default_signature = module.signatures["default"]
result = default_signature(text=tf.constant(["Hello world"]))
```

### Fine-Tuning

```python
# Load a trainable module
embedding_layer = hub.KerasLayer(
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    trainable=True,  # Enable fine-tuning
    input_shape=[],
    dtype=tf.string)

model = tf.keras.Sequential([
    embedding_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(train_data, train_labels, epochs=5)
```

### Saving Fine-Tuned Models

```python
# Save with the fine-tuned Hub weights
model.save("fine_tuned_model")

# Export as SavedModel for TF Serving
tf.saved_model.save(model, "serving_model/1")
```

---

## TensorFlow.js

### Overview

TensorFlow.js enables running machine learning models in the browser and
Node.js.

### Converting Models

```bash
# Convert SavedModel to TF.js format
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    /path/to/saved_model \
    /path/to/tfjs_model

# Convert Keras model
tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model \
    model.h5 \
    /path/to/tfjs_model

# Convert TFLite model
tensorflowjs_converter \
    --input_format=tflite \
    --output_format=tfjs_graph_model \
    model.tflite \
    /path/to/tfjs_model
```

### Loading Models in Browser

```javascript
// Load TF.js
import * as tf from '@tensorflow/tfjs';

// Load graph model
const model = await tf.loadGraphModel('https://example.com/model/model.json');

// Load layers model (Keras-style)
const model = await tf.loadLayersModel('https://example.com/model/model.json');

// Run inference
const input = tf.tensor2d([[1, 2, 3, 4]]);
const output = model.predict(input);
output.print();
```

### Backends

```javascript
// Set backend
await tf.setBackend('webgl');     // GPU via WebGL (default)
await tf.setBackend('wasm');      // WebAssembly (fast CPU)
await tf.setBackend('cpu');       // JavaScript CPU (slowest)

// Check backend
console.log(tf.getBackend());

// Wait for backend ready
await tf.ready();
```

### TF.js Node.js

```javascript
const tf = require('@tensorflow/tfjs-node');
// Or with GPU:
// const tf = require('@tensorflow/tfjs-node-gpu');

// Load SavedModel directly
const model = await tf.node.loadSavedModel('/path/to/saved_model');
const output = model.predict(input);
```

---

## Mobile Deployment

### Android (TFLite)

```java
// build.gradle
implementation 'org.tensorflow:tensorflow-lite:2.14.0'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.0'

// Java inference
try (Interpreter interpreter = new Interpreter(modelBuffer)) {
    float[][] input = new float[1][224][224][3];
    float[][] output = new float[1][numClasses];

    interpreter.run(input, output);
}
```

### Android with GPU Delegate

```java
GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
gpuOptions.setForceBackend(GpuDelegate.Options.OPENCL);
GpuDelegate gpuDelegate = new GpuDelegate(gpuOptions);

Interpreter.Options options = new Interpreter.Options();
options.addDelegate(gpuDelegate);

Interpreter interpreter = new Interpreter(modelBuffer, options);
```

### Android with NNAPI

```java
NnApiDelegate nnapiDelegate = new NnApiDelegate();
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(nnapiDelegate);
options.setUseNNAPI(true);

Interpreter interpreter = new Interpreter(modelBuffer, options);
```

### iOS (TFLite)

```swift
// Podfile
pod 'TensorFlowLiteSwift'

// Swift inference
import TensorFlowLite

let interpreter = try Interpreter(modelPath: modelPath)
try interpreter.allocateTensors()

var inputTensor = try interpreter.input(at: 0)
try inputTensor.copy(data: inputData)
try interpreter.invoke()

let outputTensor = try interpreter.output(at: 0)
let results = outputTensor.data
```

### iOS with Metal Delegate

```swift
var delegate: MetalDelegate? = nil
if let metalDelegate = MetalDelegate() {
    delegate = metalDelegate
    let options = Interpreter.Options()
    let interpreter = try Interpreter(modelPath: modelPath,
                                       delegates: [metalDelegate])
}
```

---

## Cloud Deployment

### Google Cloud AI Platform

```bash
# Upload model to Cloud Storage
gsutil cp -r serving_model/1 gs://my-bucket/models/my_model/1

# Create model
gcloud ai-platform models create my_model \
    --regions=us-central1

# Create version
gcloud ai-platform versions create v1 \
    --model=my_model \
    --origin=gs://my-bucket/models/my_model/1 \
    --runtime-version=2.14 \
    --python-version=3.9 \
    --machine-type=n1-standard-4

# Predict
gcloud ai-platform predict \
    --model=my_model \
    --version=v1 \
    --json-request=request.json
```

### AWS SageMaker

```python
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# Create model
model = TensorFlowModel(
    model_data='s3://my-bucket/model.tar.gz',
    role=sagemaker.get_execution_role(),
    framework_version='2.14',
    entry_point='inference.py')

# Deploy
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge')

# Predict
result = predictor.predict(data)
```

### Azure ML

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, OnlineEndpoint, OnlineDeployment

# Register model
model = Model(path="./serving_model", name="my-model")
ml_client.models.create_or_update(model)

# Create endpoint
endpoint = OnlineEndpoint(name="my-endpoint")
ml_client.online_endpoints.begin_create_or_update(endpoint)

# Deploy
deployment = OnlineDeployment(
    name="blue",
    endpoint_name="my-endpoint",
    model=model.id,
    instance_type="Standard_DS3_v2",
    instance_count=1)
ml_client.online_deployments.begin_create_or_update(deployment)
```

---

## Kubernetes Deployment

### TF Serving on Kubernetes

```yaml
# tf-serving-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tf-serving
  template:
    metadata:
      labels:
        app: tf-serving
    spec:
      containers:
      - name: tf-serving
        image: tensorflow/serving:latest-gpu
        ports:
        - containerPort: 8500
        - containerPort: 8501
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
        env:
        - name: MODEL_NAME
          value: "my_model"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: tf-serving
spec:
  selector:
    app: tf-serving
  ports:
  - name: grpc
    port: 8500
    targetPort: 8500
  - name: rest
    port: 8501
    targetPort: 8501
  type: LoadBalancer
```

### TF Operator (Kubeflow)

```yaml
# TFJob for distributed training
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: mnist-training
spec:
  tfReplicaSpecs:
    PS:
      replicas: 2
      template:
        spec:
          containers:
          - name: tensorflow
            image: my-training-image:latest
          restartPolicy: OnFailure
    Worker:
      replicas: 4
      template:
        spec:
          containers:
          - name: tensorflow
            image: my-training-image:latest
            resources:
              limits:
                nvidia.com/gpu: 1
          restartPolicy: OnFailure
```

### Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tf-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tf-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

---

## Edge Deployment

### TFLite on Edge Devices

```python
# Convert model for edge deployment
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model_dir")

# Optimize for edge
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Enable Edge TPU compilation
# (requires pycoral library and Edge TPU compiler)
tflite_model = converter.convert()

with open("model_edge.tflite", "wb") as f:
    f.write(tflite_model)
```

### Edge TPU (Google Coral)

```python
from pycoral.utils import edgetpu
from pycoral.adapters import common, classify

# List Edge TPUs
print(edgetpu.list_edge_tpus())

# Compile for Edge TPU
# Use: edgetpu_compiler model_edge.tflite
# Produces: model_edge_edgetpu.tflite

# Run inference
interpreter = make_interpreter("model_edge_edgetpu.tflite")
interpreter.allocate_tensors()

# Set input
common.set_input(interpreter, image)
interpreter.invoke()

# Get results
results = classify.get_classes(interpreter, top_k=5)
```

### TFLite Micro on Microcontrollers

See reference 37 (TFLite Micro) for detailed coverage of edge deployment on
microcontrollers including ARM Cortex-M, RISC-V, and Xtensa platforms.

---

## Model Optimization Toolkit

### Pruning

```python
import tensorflow_model_optimization as tfmot

# Prune during training
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Define pruning schedule
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
        0.5,  # 50% sparsity
        begin_step=0,
        frequency=100)
}

# Apply pruning to model
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Train with pruning
model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
]

model_for_pruning.fit(
    train_dataset,
    epochs=10,
    callbacks=callbacks)

# Strip pruning wrappers for export
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

### Quantization Aware Training

```python
import tensorflow_model_optimization as tfmot

# Apply quantization aware training
quantize_model = tfmot.quantization.keras.quantize_model

# Quantize the entire model
q_aware_model = quantize_model(model)

# Or quantize specific layers
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
QuantizeScope = tfmot.quantization.keras.QuantizeScope

# Custom quantization config
class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, tfmot.quantization.keras.LastValueQuantizer())]

    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, tfmot.quantization.keras.MovingAverageQuantizer())]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

# Train
q_aware_model.compile(optimizer='adam', loss='mse')
q_aware_model.fit(train_data, epochs=5)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
```

### Clustering (Weight Sharing)

```python
import tensorflow_model_optimization as tfmot

# Apply clustering
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
    'number_of_clusters': 16,
    'cluster_centroids_init': CentroidInitialization.LINEAR
}

clustered_model = cluster_weights(model, **clustering_params)

# Train
clustered_model.compile(optimizer='adam', loss='mse')
clustered_model.fit(train_data, epochs=5)

# Strip clustering wrappers
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
```

### Combined Optimization Pipeline

```python
# 1. Prune
pruned_model = prune_low_magnitude(model, **pruning_params)
pruned_model.fit(...)
pruned_model = strip_pruning(pruned_model)

# 2. Cluster
clustered_model = cluster_weights(pruned_model, **clustering_params)
clustered_model.fit(...)
clustered_model = strip_clustering(clustered_model)

# 3. Quantize
converter = tf.lite.TFLiteConverter.from_keras_model(clustered_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
optimized_model = converter.convert()
```

---

## Export Patterns

### Export for TF Serving

```python
# Export SavedModel with serving signature
model.save("serving_model/1")

# Or with explicit signature
class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 784], dtype=tf.float32)
    ])
    def serve(self, images):
        return self.model(images, training=False)

export_model = ExportModel(model)
tf.saved_model.save(export_model, "serving_model/1",
    signatures={"serving_default": export_model.serve})
```

### Export for TFLite

```python
# Basic export
converter = tf.lite.TFLiteConverter.from_saved_model("model_dir")
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Full integer quantization
def representative_dataset():
    for data in calibration_data:
        yield [data]

converter = tf.lite.TFLiteConverter.from_saved_model("model_dir")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
quantized_model = converter.convert()
```

### Export for TF.js

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    ./saved_model \
    ./tfjs_model
```

### Multi-Platform Export

```python
import tensorflow as tf

def export_for_all_platforms(model, export_dir):
    """Export model for multiple deployment targets."""

    # 1. SavedModel (TF Serving, Python)
    saved_model_dir = f"{export_dir}/saved_model/1"
    model.save(saved_model_dir)

    # 2. TFLite FP32 (mobile, edge)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    fp32_model = converter.convert()
    with open(f"{export_dir}/tflite/model_fp32.tflite", "wb") as f:
        f.write(fp32_model)

    # 3. TFLite INT8 (mobile optimized)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    int8_model = converter.convert()
    with open(f"{export_dir}/tflite/model_int8.tflite", "wb") as f:
        f.write(int8_model)

    # 4. TF.js (web)
    # Use tensorflowjs_converter CLI tool

    # 5. ONNX (optional, for cross-framework)
    # Use tf2onnx CLI tool
```

---

## Summary

TensorFlow provides a comprehensive deployment ecosystem:

1. **TF Serving**: Production-grade model serving with REST/gRPC APIs,
   versioning, batching, and Docker support.
2. **TF Hub**: Reusable model modules for transfer learning.
3. **TF.js**: Browser and Node.js deployment with WebGL/WASM backends.
4. **Mobile**: TFLite on Android/iOS with hardware acceleration delegates.
5. **Cloud**: Native support for GCP, AWS, and Azure ML platforms.
6. **Kubernetes**: TF Serving on K8s with auto-scaling and GPU support.
7. **Edge**: TFLite Micro and Edge TPU for resource-constrained devices.
8. **Optimization**: Pruning, quantization, and clustering for model
   compression.
9. **Multi-platform export**: Single model exported to multiple deployment
   targets.
