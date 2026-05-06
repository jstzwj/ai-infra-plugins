# ONNX Runtime Reference - Chapters 31-36: Language Bindings

---

## 31. C# API Reference (Microsoft.ML.OnnxRuntime)

### Installation
```bash
dotnet add package Microsoft.ML.OnnxRuntime          # CPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu       # CUDA GPU
dotnet add package Microsoft.ML.OnnxRuntime.DirectML  # DirectML
```

### InferenceSession
```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Create session
var session = new InferenceSession("model.onnx");

// With options
var options = new SessionOptions();
options.InterOpNumThreads = 4;
options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
options.AppendExecutionProvider_CPU(1);
var session = new InferenceSession("model.onnx", options);

// Run inference
var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
var input = new List<NamedOnnxValue> {
    NamedOnnxValue.CreateFromTensor("input", inputTensor)
};
using var results = session.Run(input);

// Get output
var output = results.First().AsTensor<float>();
Console.WriteLine($"Output shape: [{string.Join(",", output.Dimensions)}]");

// Async inference
var task = Task.Run(() => session.Run(input));
var results = await task;

// Model info
foreach (var meta in session.InputMetadata)
    Console.WriteLine($"Input: {meta.Key}, Shape: [{string.Join(",", meta.Value.Dimensions)}]");
foreach (var meta in session.OutputMetadata)
    Console.WriteLine($"Output: {meta.Key}, Shape: [{string.Join(",", meta.Value.Dimensions)}]");

// Model metadata
var modelMeta = session.ModelMetadata;
Console.WriteLine($"Producer: {modelMeta.ProducerName}");
Console.WriteLine($"Graph: {modelMeta.GraphName}");
Console.WriteLine($"Description: {modelMeta.Description}");
Console.WriteLine($"Version: {modelMeta.Version}");

// Error handling
try {
    var results = session.Run(input);
} catch (OnnxRuntimeException e) {
    Console.WriteLine($"ORT Error: {e.Message}");
}
```

### SessionOptions
```csharp
var options = new SessionOptions();
options.InterOpNumThreads = 4;
options.IntraOpNumThreads = 1;
options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
options.EnableMemoryPattern = true;
options.EnableProfiling = true;
options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

// CUDA EP
options.AppendExecutionProvider_CUDA(0);

// DirectML EP
options.AppendExecutionProvider_DML(0);

// Custom ops
options.RegisterCustomOpLibrary("my_ops.dll");

// Config entries
options.AddSessionConfigEntry("session.disable_prepacking", "0");
```

### Tensor Types
```csharp
// DenseTensor
var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
tensor.Fill(1.0f);

// From array
var data = new float[1 * 3 * 224 * 224];
var tensor = new DenseTensor<float>(data, new[] { 1, 3, 224, 224 });

// Access data
float value = tensor[0, 0, 0, 0];
tensor[0, 0, 0, 0] = 2.0f;

// NamedOnnxValue
var namedValue = NamedOnnxValue.CreateFromTensor("input", tensor);
var name = namedValue.Name;
var tensorOut = namedValue.AsTensor<float>();
```

---

## 32. Java API Reference

### Maven Dependency
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.22.0</version>
</dependency>
```

### Usage
```java
import ai.onnxruntime.*;

// Create environment
OrtEnvironment env = OrtEnvironment.getEnvironment();

// Create session
OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
opts.setInterOpNumThreads(4);

OrtSession session = env.createSession("model.onnx", opts);

// Create input tensor
float[] inputData = new float[3 * 224 * 224];
long[] shape = {1, 3, 224, 224};
OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

// Run inference
Map<String, OnnxTensor> inputs = new HashMap<>();
inputs.put("input", inputTensor);
try (OrtSession.Result results = session.run(inputs)) {
    OnnxTensor output = (OnnxTensor) results.get(0);
    float[] outputData = output.getFloatBuffer().array();
    long[] outputShape = output.getInfo().getShape();
}

// Get model info
for (NodeInfo info : session.getInputInfo().values()) {
    System.out.println("Input: " + info.getName() + " Shape: " + info.getInfo().getShape());
}

// Cleanup
session.close();
env.close();
```

---

## 33. JavaScript/TypeScript API Reference

### Installation
```bash
# Node.js
npm install onnxruntime-node

# Browser
npm install onnxruntime-web
```

### Node.js Usage
```javascript
const ort = require('onnxruntime-node');

async function main() {
    // Create session
    const session = await ort.InferenceSession.create('model.onnx', {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
    });

    // Create input tensor
    const input = new ort.Tensor('float32', new Float32Array(3 * 224 * 224), [1, 3, 224, 224]);

    // Run inference
    const results = await session.run({ input });
    const output = results.output;
    console.log('Output shape:', output.dims);
}

main();
```

### Browser Usage
```javascript
import * as ort from 'onnxruntime-web';

async function main() {
    const session = await ort.InferenceSession.create('model.onnx', {
        executionProviders: ['webgpu', 'wasm'],
    });

    const input = new ort.Tensor('float32', new Float32Array(3 * 224 * 224), [1, 3, 224, 224]);
    const results = await session.run({ input });
}
```

### Tensor API
```javascript
// Create tensor
const tensor = new ort.Tensor('float32', new Float32Array([1, 2, 3, 4]), [2, 2]);

// Types: 'float32', 'float64', 'int8', 'int16', 'int32', 'int64',
//        'uint8', 'uint16', 'uint32', 'uint64', 'bool', 'string'

// Access data
console.log(tensor.data);    // TypedArray
console.log(tensor.dims);    // [2, 2]
console.log(tensor.type);    // 'float32'
console.log(tensor.size);    // 4
```

---

## 34. WebAssembly Deployment

### Build
```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest

# Build ORT for WASM
python build.sh --config Release --build_wasm --parallel \
    --target onnxruntime_webassembly
```

### Web Deployment
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script>
    async function runInference() {
        const session = await ort.InferenceSession.create('model.onnx', {
            executionProviders: ['wasm'],
        });

        const input = new ort.Tensor('float32', new Float32Array(3*224*224), [1,3,224,224]);
        const results = await session.run({ input });
    }
</script>
```

---

## 35. Rust API Reference

```rust
use onnxruntime::{environment::Environment, session::Session, tensor::OrtOwnedTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = Environment::builder().with_name("test").build()?;

    let mut session = Session::builder()?.with_model_from_file("model.onnx")?;

    let input = ndarray::Array::from_shape_vec((1, 3, 224, 224), vec![1.0f32; 3*224*224])?;

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run([input])?;

    Ok(())
}
```

---

## 36. Objective-C API Reference

```objc
#import <onnxruntime_objc/ort_session.h>

// Create session
ORTSessionOptions *opts = [[ORTSessionOptions alloc] initWithError:&error];
[opts setIntraOpNumThreads:4];

ORTSession *session = [[ORTSession alloc] initWithModelPath:@"model.onnx"
                                                sessionOptions:opts
                                                        error:&error];

// Create input
ORTValue *input = [ORTValue tensorWithData:inputData
                                     shape:@[@1, @3, @224, @224]
                                  elementType:ORTTensorElementTypeFloat
                                      error:&error];

// Run inference
NSDictionary *outputs = [session runWithInputs:@{@"input": input}
                                  outputNames:[NSSet setWithArray:@[@"output"]]
                                       error:&error];
```
