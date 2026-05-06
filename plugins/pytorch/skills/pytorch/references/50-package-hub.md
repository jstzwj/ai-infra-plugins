# torch.package and torch.hub

## Overview

PyTorch provides two systems for model distribution and deployment:

1. **torch.package**: A packaging system that bundles a model with all its code dependencies into a single archive for deployment. It ensures the model can be loaded and run in a different environment without requiring the original codebase.

2. **torch.hub**: A model zoo for sharing pre-trained models hosted on GitHub repositories. Users can discover, download, and use models with a single API call.

---

## torch.package

### Overview

`torch.package` creates self-contained model archives that include:
- Model parameters (state_dict)
- Model code (Python source files)
- Dependent modules and resources
- Pickled Python objects

The resulting package is a ZIP archive that can be loaded on any machine with PyTorch installed, without needing the original source code.

**Source location**: `torch/package/`

### PackageExporter

```python
import torch
import torch.package

torch.package.PackageExporter(
    file: Union[str, Path, BinaryIO],
    export_dependencies: bool = True,
    debug: bool = True,
    pickle_protocol: int = 4,
    config: Optional[PackageConfig] = None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | `str` or `Path` or `BinaryIO` | Output file path or file-like object |
| `export_dependencies` | `bool` | Automatically export dependencies |
| `debug` | `bool` | Include debug information |
| `pickle_protocol` | `int` | Pickle protocol version (4 for compatibility) |
| `config` | `PackageConfig` | Package configuration |

### Exporter Methods

#### save_module

```python
exporter.save_module(
    module_name: str,
    deps: Optional[Set[str]] = None,
    *args,
    **kwargs,
) -> None
```

Save a Python module and its dependencies into the package.

```python
import torch.package

with torch.package.PackageExporter("model_package.pt") as exporter:
    # Save the model's module
    exporter.save_module("myapp.models.resnet")
    exporter.save_module("myapp.models.layers")
```

#### save_pickle

```python
exporter.save_pickle(
    name: str,
    pickle_name: str,
    obj: Any,
    dependencies: Optional[Set[str]] = None,
) -> None
```

Save a Python object using pickle.

```python
with torch.package.PackageExporter("model.pt") as exporter:
    model = MyModel()
    model.load_state_dict(torch.load("weights.pth"))

    # Save the model object itself (not just the module)
    exporter.save_pickle("models", "my_model.pkl", model)

    # Save additional objects
    exporter.save_pickle("config", "model_config.pkl", model.config)
    exporter.save_pickle("data", "label_map.pkl", label_mapping)
```

#### save_source_file

```python
exporter.save_source_file(
    module_name: str,
    file_path: str,
    dependencies: Optional[Set[str]] = None,
) -> None
```

Save a specific source file into the package.

```python
with torch.package.PackageExporter("model.pt") as exporter:
    exporter.save_source_file(
        "myapp.custom_ops",
        "/path/to/custom_ops.py",
    )
```

#### save_text and save_binary

```python
# Save text content
exporter.save_text(
    name: str,          # resource name
    text: str,          # text content
)

# Save binary content
exporter.save_binary(
    name: str,          # resource name
    binary: bytes,      # binary content
)

# Usage
with torch.package.PackageExporter("model.pt") as exporter:
    exporter.save_text("config/model_config.json", json.dumps(config))
    exporter.save_binary("data/vocab.bin", vocab_bytes)
    exporter.save_text("README", "Model package for inference")
```

#### extern_module

```python
exporter.extern_module(
    module_name: str,
    dependents: Optional[Set[str]] = None,
) -> None
```

Mark a module as external -- it will NOT be packaged and must be available on the target system.

```python
with torch.package.PackageExporter("model.pt") as exporter:
    # External modules: use system-installed versions
    exporter.extern_module("torch")
    exporter.extern_module("numpy")
    exporter.extern_module("PIL")
    exporter.extern_module("cv2")

    # External a custom library that must be pre-installed
    exporter.extern_module("my_custom_cuda_lib")
```

#### mock_module

```python
exporter.mock_module(
    module_name: str,
    dependents: Optional[Set[str]] = None,
) -> None
```

Replace a module with a mock that raises an error if used at runtime. Useful for modules that are imported but not actually needed.

```python
with torch.package.PackageExporter("model.pt") as exporter:
    # Mock modules that are imported but not used
    exporter.mock_module("matplotlib")  # only used for training visualization
    exporter.mock_module("tensorboard")  # only used for training logging
    exporter.mock_module("tqdm")         # only used for progress bars
```

#### deny_module

```python
exporter.deny_module(
    module_name: str,
    dependents: Optional[Set[str]] = None,
) -> None
```

Prevent a module from being included. Loading will fail if the module is needed.

```python
with torch.package.PackageExporter("model.pt") as exporter:
    # Deny modules that should never be included
    exporter.deny_module("training_utils")
    exporter.deny_module("test_utils")
```

---

## PackageImporter

```python
import torch.package

torch.package.PackageImporter(
    file: Union[str, Path, BinaryIO],
    map_location: Optional[Union[str, torch.device]] = None,
    config: Optional[PackageConfig] = None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | `str` or `Path` or `BinaryIO` | Package file path or file-like object |
| `map_location` | `str` or `device` | Device to load tensors to |
| `config` | `PackageConfig` | Import configuration |

### Importer Methods

#### load_module

```python
importer.load_module(
    module_name: str,
) -> ModuleType
```

Load a Python module from the package.

```python
importer = torch.package.PackageImporter("model.pt")

# Load a module from the package
models_module = importer.load_module("myapp.models.resnet")
model_class = models_module.ResNet18
model = model_class()
```

#### load_pickle

```python
importer.load_pickle(
    name: str,
    pickle_name: str,
) -> Any
```

Load a pickled object from the package.

```python
importer = torch.package.PackageImporter("model.pt")

# Load the model
model = importer.load_pickle("models", "my_model.pkl")

# Load config
config = importer.load_pickle("config", "model_config.pkl")

# Load label mapping
labels = importer.load_pickle("data", "label_map.pkl")

# Run inference
output = model(input_tensor)
```

#### load_text and load_binary

```python
# Load text
text = importer.load_text("config/model_config.json")

# Load binary
vocab = importer.load_binary("data/vocab.bin")
```

---

## Dependency Analysis and Management

### Automatic Dependency Analysis

When `export_dependencies=True` (default), the exporter automatically:
1. Traces all imports used by saved modules
2. Analyzes class hierarchies and function references
3. Includes all transitively imported modules

### Dependency Resolution Strategy

```python
# The exporter resolves dependencies in this order:
# 1. Check if module is extern'd -> use system version
# 2. Check if module is mock'd -> use mock stub
# 3. Check if module is deny'd -> error
# 4. Check if module is already saved -> skip
# 5. Package the module's source code

with torch.package.PackageExporter("model.pt") as exporter:
    # Explicitly control dependency resolution
    exporter.extern_module("torch")       # use installed PyTorch
    exporter.extern_module("numpy")       # use installed numpy
    exporter.mock_module("matplotlib")    # mock unused deps

    # Save model (auto-analyzes dependencies)
    exporter.save_pickle("models", "model.pkl", model)
```

### Package Structure

A `.pt` package is a ZIP archive with this structure:

```
model.pt/
  +-- models/
  |     +-- my_model.pkl        # pickled model
  |     +-- model_config.pkl    # pickled config
  +-- data/
  |     +-- label_map.pkl       # pickled data
  |     +-- vocab.bin           # binary resource
  +-- config/
  |     +-- model_config.json   # text resource
  +-- myapp/
  |     +-- models/
  |     |     +-- __init__.py
  |     |     +-- resnet.py     # packaged source
  |     |     +-- layers.py     # packaged source
  |     +-- utils/
  |           +-- __init__.py
  |           +-- transforms.py
  +-- .package/
        +-- MANIFEST            # package manifest
        +-- extern_modules      # list of extern'd modules
        +-- mock_modules        # list of mock'd modules
```

---

## Packaging Constraints

### What Can Be Packaged

```python
# Supported:
# - torch.nn.Module instances
# - torch.Tensor objects
# - Standard Python types (dict, list, tuple, etc.)
# - Custom Python classes (source code is packaged)
# - NumPy arrays
# - Functions and lambdas

# Not supported / requires special handling:
# - C extensions (must be extern'd)
# - Native code (.so, .dll, .dylib files)
# - Modules that depend on system libraries
# - Modules that use __file__ for resource loading
# - Dynamic imports (importlib.import_module with variable names)
```

### Common Issues and Solutions

```python
# Issue: Module uses __file__ to load resources
# Solution: Use package_resource or save files explicitly
import importlib.resources as pkg_resources

# Issue: C extension not found
# Solution: extern_module the C extension
exporter.extern_module("torch._C")
exporter.extern_module("my_custom_kernel")

# Issue: Dynamic import not detected
# Solution: Explicitly save the module
exporter.save_module("myapp.dynamic_module")

# Issue: Global state not captured
# Solution: Save state explicitly
exporter.save_pickle("state", "globals.pkl", my_global_state)
```

---

## torch.hub

### Overview

`torch.hub` provides a simple API for sharing and loading pre-trained models from GitHub repositories. Model authors publish their models by adding a `hubconf.py` file to their repository.

### torch.hub.list

```python
torch.hub.list(
    github: str,                    # "username/repo[:tag]"
    force_reload: bool = False,     # force reload from remote
    trust_repo: Optional[bool] = None,  # trust the repository
    verbose: bool = True,
) -> List[str]
```

List available models in a GitHub repository.

```python
import torch.hub

# List models in PyTorch Vision
models = torch.hub.list("pytorch/vision:v0.10.0", force_reload=False)
print(models)
# ['alexnet', 'deeplabv3_resnet50', 'densenet121', 'googlenet', 'inception_v3',
#  'mnasnet0_5', 'mobilenet_v2', 'resnet18', 'resnext50_32x4d', 'vgg11', ...]
```

### torch.hub.load

```python
torch.hub.load(
    repo_or_dir: Union[str, Path],  # GitHub repo or local directory
    model: str,                     # model function name from hubconf.py
    *args,                          # positional arguments for the model function
    source: str = "github",         # "github" or "local"
    force_reload: bool = False,
    trust_repo: Optional[bool] = None,
    verbose: bool = True,
    skip_validation: bool = False,
    **kwargs,                       # keyword arguments for the model function
) -> Any
```

Load a pre-trained model from a hub repository.

```python
import torch
import torch.hub

# Load a pre-trained ResNet
model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet18",
    pretrained=True,
)
model.eval()

# Use the model
output = model(torch.randn(1, 3, 224, 224))

# Load with specific arguments
model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet50",
    pretrained=True,
    progress=True,  # show download progress
)

# Load from a local directory (for development)
model = torch.hub.load(
    "/path/to/local/vision",
    "resnet18",
    source="local",
    pretrained=True,
)
```

### torch.hub.load_state_dict_from_url

```python
torch.hub.load_state_dict_from_url(
    url: str,                                   # URL to state_dict file
    model_dir: Optional[str] = None,            # directory to save downloaded file
    map_location: Optional[Union[str, torch.device]] = None,
    progress: bool = True,                      # show download progress
    check_hash: bool = False,                   # verify file hash
    file_name: Optional[str] = None,            # custom file name
) -> Dict[str, Any]
```

Download and load a model's state_dict from a URL.

```python
import torch

# Load state_dict from URL
state_dict = torch.hub.load_state_dict_from_url(
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    map_location="cpu",
    progress=True,
    check_hash=True,
)

# Apply to model
model = torchvision.models.resnet18()
model.load_state_dict(state_dict)

# Custom download directory
state_dict = torch.hub.load_state_dict_from_url(
    "https://my-server.com/models/my_model.pth",
    model_dir="/path/to/models/",
    file_name="my_model_v1.pth",
)
```

### torch.hub.set_dir and torch.hub.get_dir

```python
# Get current hub cache directory
hub_dir = torch.hub.get_dir()
# Default: ~/.cache/torch/hub

# Set custom hub cache directory
torch.hub.set_dir("/shared/models/torch_hub")

# The hub directory stores:
# /shared/models/torch_hub/
#   +-- pytorch_vision_v0.10.0/    # cloned repos
#   +-- pytorch_vision_v0.10.0.zip # downloaded snapshots
#   +-- checkpoints/               # downloaded weights
```

---

## Trust Repository Management

### Trust Settings

```python
# torch.hub prompts for trust when loading from a new repository
# This can be controlled:

# Trust a specific repository permanently
torch.hub.set_trust_repo("pytorch/vision", True)

# Load with explicit trust
model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet18",
    trust_repo=True,  # skip trust prompt
)

# Deny trust
model = torch.hub.load(
    "unknown/repo:main",
    "model",
    trust_repo=False,  # will fail for untrusted repos
)
```

---

## Creating a hubconf.py

### Basic Structure

```python
# hubconf.py - place in the root of your GitHub repository

import torch

# Optional dependencies that might not be available
dependencies = ["torch", "torchvision"]

def resnet18(pretrained=False, **kwargs):
    """Load a pre-trained ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar during download
    """
    from torchvision.models.resnet import ResNet, BasicBlock

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            progress=True,
        )
        model.load_state_dict(state_dict)

    return model

def resnet50(pretrained=False, **kwargs):
    """Load a pre-trained ResNet-50 model."""
    from torchvision.models.resnet import ResNet, Bottleneck

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            progress=True,
        )
        model.load_state_dict(state_dict)

    return model
```

### Advanced hubconf.py

```python
# hubconf.py with multiple model configurations

dependencies = ["torch"]

def my_model(
    model_name: str = "base",
    pretrained: bool = False,
    num_classes: int = 1000,
    device: str = "cpu",
    **kwargs,
):
    """
    Available models:
        - "base": Base model (50M parameters)
        - "large": Large model (200M parameters)
        - "tiny": Tiny model (5M parameters)
    """
    from mylib.models import ModelBase, ModelLarge, ModelTiny

    models = {
        "base": ModelBase,
        "large": ModelLarge,
        "tiny": ModelTiny,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    model = models[model_name](num_classes=num_classes, **kwargs)

    if pretrained:
        checkpoint_urls = {
            "base": "https://my-server.com/checkpoints/base_v1.pth",
            "large": "https://my-server.com/checkpoints/large_v1.pth",
            "tiny": "https://my-server.com/checkpoints/tiny_v1.pth",
        }
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_urls[model_name],
            map_location=device,
        )
        model.load_state_dict(state_dict)

    return model.to(device)
```

---

## Example: Package a Model for Deployment

```python
import torch
import torch.package
import json

# Define model
class MyModel(torch.nn.Module):
    def __init__(self, vocab_size=10000, hidden_dim=256, num_classes=10):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_dim, nhead=8),
            num_layers=4,
        )
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x)

# Train or load model
model = MyModel()
# ... training code ...
model.eval()

# Package the model
with torch.package.PackageExporter("deployed_model.pt") as exporter:
    # External modules (must be available on deployment machine)
    exporter.extern_module("torch")
    exporter.extern_module("numpy")

    # Mock modules not needed for inference
    exporter.mock_module("matplotlib")
    exporter.mock_module("tensorboard")
    exporter.mock_module("tqdm")
    exporter.mock_module("sklearn")

    # Save the model
    exporter.save_pickle("model", "model.pkl", model)

    # Save configuration
    config = {
        "vocab_size": 10000,
        "hidden_dim": 256,
        "num_classes": 10,
        "version": "1.0.0",
    }
    exporter.save_text("config", "model_config.json", json.dumps(config, indent=2))

    # Save label mapping
    labels = {0: "positive", 1: "negative", 2: "neutral"}
    exporter.save_pickle("data", "labels.pkl", labels)

print("Model packaged successfully: deployed_model.pt")
```

### Load and Use Packaged Model

```python
import torch.package

# Load the package
importer = torch.package.PackageImporter(
    "deployed_model.pt",
    map_location="cpu",
)

# Load model
model = importer.load_pickle("model", "model.pkl")
model.eval()

# Load config
import json
config = json.loads(importer.load_text("config", "model_config.json"))

# Load labels
labels = importer.load_pickle("data", "labels.pkl")

# Run inference
input_ids = torch.randint(0, 10000, (1, 128))
with torch.no_grad():
    logits = model(input_ids)
    predicted_class = logits.argmax(dim=-1).item()
    predicted_label = labels[predicted_class]

print(f"Predicted: {predicted_label}")
print(f"Config: {config}")
```

---

## Example: Load and Use a Hub Model

```python
import torch
import torch.hub

# List available models
models = torch.hub.list("pytorch/vision:v0.10.0")
print(f"Available models: {models}")

# Load pre-trained model
model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet18",
    pretrained=True,
)
model.eval()

# Prepare input
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and classify an image
from PIL import Image
img = Image.open("test_image.jpg")
input_tensor = preprocess(img).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)

# Get top predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
for prob, catid in zip(top5_prob, top5_catid):
    print(f"Class {catid.item()}: {prob.item():.4f}")

# Cache management
print(f"Hub cache: {torch.hub.get_dir()}")

# Force reload (useful after repo updates)
model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "resnet18",
    pretrained=True,
    force_reload=True,
)
```

---

## Comparison with TorchScript Serialization

| Feature | torch.package | TorchScript |
|---------|--------------|-------------|
| **Format** | ZIP archive with source code | Serialized binary |
| **Code required** | Yes (Python source included) | No (compiled to IR) |
| **Python dependency** | Requires Python runtime | Can run in C++ (LibTorch) |
| **Custom ops** | Full support (source included) | Limited (must be registered) |
| **Dynamic control flow** | Full support | Limited (must be scriptable) |
| **File size** | Larger (includes source) | Smaller (compiled) |
| **Cross-language** | Python only | Python + C++ + mobile |
| **Model authoring** | Any Python code | Must be TorchScript-compatible |
| **Use case** | Python deployment, model sharing | Production, mobile, C++ serving |

### When to Use Which

```python
# Use torch.package when:
# - Deploying to Python environments
# - Model has complex Python logic
# - Need to include custom ops
# - Want to share models with researchers

# Use TorchScript when:
# - Deploying to C++ / mobile
# - Need smallest possible model size
# - Running in LibTorch without Python
# - Production serving with strict latency requirements

# Use torch.hub when:
# - Sharing pre-trained models publicly
# - Quick prototyping with standard models
# - No custom packaging needed
```

### Combining Both

```python
import torch
import torch.package

# Package a TorchScript model
model = torch.jit.script(MyModel())

with torch.package.PackageExporter("package.pt") as exporter:
    exporter.extern_module("torch")
    exporter.save_pickle("scripted", "model.pkl", model)

# Load and use
importer = torch.package.PackageImporter("package.pt")
scripted_model = importer.load_pickle("scripted", "model.pkl")
output = scripted_model(input_tensor)
```

---

## Summary

PyTorch's model distribution tools provide:

1. **torch.package.PackageExporter**: Creates self-contained archives with model code and data
2. **PackageExporter methods**: save_module, save_pickle, save_source_file, save_text, save_binary, extern_module, mock_module, deny_module
3. **torch.package.PackageImporter**: Loads packages with load_module, load_pickle, load_text, load_binary
4. **Dependency management**: Automatic analysis with extern/mock/deny controls
5. **torch.hub.list**: Discover models in GitHub repositories
6. **torch.hub.load**: Download and instantiate pre-trained models
7. **torch.hub.load_state_dict_from_url**: Download model weights from URLs
8. **hubconf.py**: Standard format for publishing models to torch.hub
9. **Trust management**: Security controls for loading external code
10. **Comparison**: torch.package for Python deployment, TorchScript for C++/mobile, torch.hub for sharing
