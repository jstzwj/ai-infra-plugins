# LibTorch C++ API

## Overview

LibTorch is PyTorch's C++ frontend, providing a complete C++ interface for tensor computation, automatic differentiation, neural network construction, optimization, and model deployment. It wraps the ATen C++ tensor library and provides higher-level abstractions that mirror the Python API.

**Source location**: `torch/csrc/api/`

**Distribution**: Pre-built binaries available from pytorch.org for Linux, macOS, and Windows (CPU and CUDA variants).

---

## Installation and Setup

### Downloading Pre-built Binaries

```bash
# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.0%2Bcpu.zip

# Download LibTorch (CUDA 12.4 version)
wget https://download.pytorch.org/libtorch/cu124/libtorch-shared-with-deps-2.5.0%2Bcu124.zip

# Extract
unzip libtorch-shared-with-deps-*.zip
```

### Directory Structure of LibTorch Distribution

```
libtorch/
  bin/               # Runtime DLLs (Windows)
  include/
    ATen/            # ATen headers
    c10/             # c10 core library headers
    torch/           # LibTorch C++ API headers
      cable/         # Cable (serialization)
      data/          # Data loading headers
      detail/        # Internal detail headers
      nn/            # Neural network module headers
      optim/         # Optimizer headers
      serialize/     # Serialization headers
      types.h        # Common types
      utils.h        # Utility functions
    torch.h          # Main umbrella header
  lib/               # Shared libraries (.so/.dylib/.dll)
    libtorch.so
    libtorch_cpu.so
    libtorch_cuda.so
    libc10.so
    libtorch_global_deps.so
  share/
    cmake/           # CMake configuration files
      Torch/
        TorchConfig.cmake
        TorchTargets.cmake
```

### CMakeLists.txt Configuration

```cmake
cmake_minimum_required(VERSION 3.18)
project(torch_example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch (set CMAKE_PREFIX_PATH to libtorch directory)
set(CMAKE_PREFIX_PATH "/path/to/libtorch")
find_package(Torch REQUIRED)

# Enable CUDA if available
# find_package(CUDA REQUIRED)

# Add executable
add_executable(example example.cpp)

# Link LibTorch
target_link_libraries(example "${TORCH_LIBRARIES}")

# Set properties
set_property(TARGET example PROPERTY CXX_STANDARD 17)

# Copy libraries to output directory (Windows)
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS} $<TARGET_FILE_DIR:example>)
endif()
```

### Minimal Build Commands

```bash
# Configure
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build
cmake --build . --config Release

# Run
./example
```

---

## Tensors

### Main Header

```cpp
#include <torch/torch.h>
// This single header includes:
// - ATen tensor operations
// - torch::Tensor and creation functions
// - torch::nn modules
// - torch::optim optimizers
// - torch::data data loading
```

### Tensor Creation

```cpp
// === From scratch ===

// zeros
torch::Tensor t1 = torch::zeros({3, 4});
torch::Tensor t1f = torch::zeros({3, 4}, torch::kFloat64);

// ones
torch::Tensor t2 = torch::ones({2, 3});

// empty (uninitialized)
torch::Tensor t3 = torch::empty({5, 5});

// full
torch::Tensor t4 = torch::full({3, 3}, 7.0);

// eye (identity)
torch::Tensor t5 = torch::eye(4);

// arange
torch::Tensor t6 = torch::arange(10);           // {0, 1, ..., 9}
torch::Tensor t7 = torch::arange(0, 10, 2);     // {0, 2, 4, 6, 8}

// linspace
torch::Tensor t8 = torch::linspace(0, 1, 5);    // {0, 0.25, 0.5, 0.75, 1}

// logspace
torch::Tensor t9 = torch::logspace(0, 2, 5);    // {1, ~3.16, 10, ~31.6, 100}

// Random
torch::Tensor t10 = torch::rand({3, 4});         // Uniform [0, 1)
torch::Tensor t11 = torch::randn({3, 4});        // Standard normal
torch::Tensor t12 = torch::randint(0, 10, {3});  // Random int [0, 10)
torch::Tensor t13 = torch::randperm(10);         // Random permutation

// === From data ===

// torch::tensor (infers type from C++ type)
torch::Tensor t14 = torch::tensor(3.14);          // scalar
torch::Tensor t15 = torch::tensor({1, 2, 3});     // 1D int
torch::Tensor t16 = torch::tensor({1.0, 2.0, 3.0}); // 1D float
torch::Tensor t17 = torch::tensor({{1, 2}, {3, 4}}); // 2D

// With explicit options
torch::Tensor t18 = torch::tensor({1, 2, 3}, torch::kFloat32);

// torch::from_blob (wraps existing memory)
float data[] = {1, 2, 3, 4, 5, 6};
torch::Tensor t19 = torch::from_blob(data, {2, 3});
// Note: data must outlive the tensor unless copied

// from_blob with custom deleter
auto* ptr = new float[12];
torch::Tensor t20 = torch::from_blob(ptr, {3, 4}, [](void* p) {
    delete[] static_cast<float*>(p);
});

// from_blob with strides
torch::Tensor t21 = torch::from_blob(data, {2, 3}, {3, 1}, torch::kFloat32);

// === From other tensors ===

torch::Tensor t22 = torch::zeros_like(t1);   // same shape, filled with 0
torch::Tensor t23 = torch::ones_like(t1);    // same shape, filled with 1
torch::Tensor t24 = torch::rand_like(t1);    // same shape, uniform random
torch::Tensor t25 = torch::randn_like(t1);   // same shape, normal random
torch::Tensor t26 = torch::empty_like(t1);   // same shape, uninitialized
```

### Tensor Indexing

```cpp
torch::Tensor t = torch::randn({3, 4, 5});

// Basic indexing
t[0];                    // first element along dim 0 -> {4, 5}
t[0][1];                 // -> {5}
t[0][1][2];              // -> scalar tensor

// Integer indexing
t.index({0});            // same as t[0]
t.index({0, 1});         // same as t[0][1]
t.index({0, 1, 2});      // same as t[0][1][2]

// Slice indexing
t.index({torch::indexing::Slice(0, 2)});    // rows 0-1 -> {2, 4, 5}
t.index({"...", 0});                        // last dim index 0 -> {3, 4}
t.index({0, torch::indexing::Slice(), 0});  // -> {4}

// Advanced indexing
auto mask = t > 0;
auto positive = t.index({mask});  // all positive values, 1D

// Using index_put
t.index_put_({mask}, 0);  // set positive values to 0

// Integer tensor indexing
auto indices = torch::tensor({0, 2});
t.index({indices});  // gather rows 0 and 2

// Select, narrow, slice
t.select(0, 1);     // select dim 0, index 1 -> {4, 5}
t.narrow(0, 0, 2);  // dim 0, start 0, length 2 -> {2, 4, 5}
t.slice(0, 0, 2);   // dim 0, start 0, end 2 -> {2, 4, 5}
```

### Tensor Operations

```cpp
torch::Tensor a = torch::randn({3, 4});
torch::Tensor b = torch::randn({3, 4});

// Arithmetic
auto c = a + b;         // addition
auto d = a - b;         // subtraction
auto e = a * b;         // element-wise multiplication
auto f = a / b;         // element-wise division

// With scalar
auto g = a + 5;
auto h = a * 2.0;
auto i = torch::pow(a, 2);

// In-place
a += b;
a *= 2;
a.add_(b);
a.mul_(5);

// Matrix operations
auto W = torch::randn({4, 5});
auto matmul_result = torch::mm(a, W);  // {3, 5}

// Comparison
auto eq = a == b;
auto gt = a > 0;

// Reductions
a.sum();
a.mean();
a.max();
a.min();
a.sum(0);            // along dim 0 -> {4}
a.mean(1, true);     // along dim 1, keepdim -> {3, 1}

// Shape manipulation
auto reshaped = a.reshape({4, 3});
auto viewed = a.view({12});
auto permuted = a.permute({1, 0});  // transpose for 2D: a.t()
auto transposed = a.transpose(0, 1);

// Concatenation and stacking
auto cat_result = torch::cat({a, b}, 0);  // along dim 0 -> {6, 4}
auto stacked = torch::stack({a, b}, 0);   // new dim -> {2, 3, 4}

// Split
auto chunks = torch::chunk(a, 2, 0);  // 2 chunks along dim 0
auto splits = torch::split(a, 2, 0);  // size 2 along dim 0
```

---

## torch::Tensor Methods

### Shape and Metadata

```cpp
torch::Tensor t = torch::randn({2, 3, 4});

// Shape
t.sizes();           // IntArrayRef: {2, 3, 4}
t.size(0);           // 2
t.size(1);           // 3
t.dim();             // 3 (number of dimensions)
t.numel();           // 24 (total elements)

// Strides
t.strides();         // IntArrayRef: {12, 4, 1}
t.stride(0);         // 12
t.stride(1);         // 4

// Type and device
t.dtype();           // torch::kFloat32
t.device();          // c10::Device(c10::kCPU)
t.layout();          // torch::kStrided

// Properties
t.is_contiguous();   // true
t.is_complex();      // false
t.is_floating_point(); // true

// Storage
t.storage_offset();  // 0

// Gradient
t.requires_grad();   // false
t.grad();            // empty tensor (no grad yet)
t.is_leaf();         // true
```

### Type Conversion and Device Transfer

```cpp
torch::Tensor t = torch::randn({3, 4});

// Type conversion
auto t64 = t.to(torch::kFloat64);
auto t32 = t.to(torch::kFloat32);
auto t16 = t.to(torch::kFloat16);
auto bf16 = t.to(torch::kBFloat16);
auto ti32 = t.to(torch::kInt32);
auto ti64 = t.to(torch::kInt64);

// Device transfer
auto cuda_t = t.to(torch::kCUDA);
auto cpu_t = cuda_t.to(torch::kCPU);
auto cuda_t0 = t.to(torch::Device(torch::kCUDA, 0));

// Combined
auto cuda_f64 = t.to(torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

// Convenience methods
auto c = t.cuda();              // -> CUDA
auto p = t.cpu();               // -> CPU
auto f = t.toType(torch::kFloat64); // type cast

// Contiguous
auto contig = t.contiguous();   // ensure contiguous memory layout

// Clone (deep copy)
auto cloned = t.clone();
```

### View and Reshape Operations

```cpp
torch::Tensor t = torch::arange(24).view({2, 3, 4});

// view (shares memory, requires contiguous)
auto v1 = t.view({6, 4});
auto v2 = t.view({-1});          // flatten -> {24}
auto v3 = t.view({2, -1});       // {2, 12}

// reshape (may copy if not contiguous)
auto r1 = t.reshape({8, 3});
auto r2 = t.reshape({-1, 4});    // {6, 4}

// squeeze / unsqueeze
auto s1 = t.squeeze();            // remove all size-1 dims
auto s2 = t.squeeze(0);           // remove dim 0 if size 1
auto u1 = t.unsqueeze(0);         // add dim at position 0

// flatten
auto flat = t.flatten();          // {24}
auto flat2 = t.flatten(1);        // {2, 12} (flatten from dim 1)
auto flat3 = t.flatten(1, 2);     // {2, 12} (flatten dims 1-2)

// expand (broadcast without copying)
auto t2 = torch::randn({1, 3});
auto expanded = t2.expand({4, 3}); // {4, 3} (broadcast dim 0)

// permute
auto p1 = t.permute({2, 0, 1});   // {4, 2, 3}

// transpose
auto tr = t.transpose(0, 1);      // swap dims 0 and 1 -> {3, 2, 4}
```

---

## Automatic Differentiation

### Basic Autograd

```cpp
#include <torch/torch.h>

// Enable gradient tracking
torch::Tensor x = torch::randn({3, 4}, torch::requires_grad());

// Forward pass
torch::Tensor y = x * 2 + 1;
torch::Tensor z = y.sum();

// Backward pass
z.backward();

// Access gradients
torch::Tensor grad = x.grad();
// grad should be all 2s (dz/dx = 2)

// Detach from computation graph
torch::Tensor detached = y.detach();
detached.requires_grad();  // false

// No gradient context
{
    torch::NoGradGuard no_grad;
    auto t = torch::randn({3, 4});
    t.requires_grad();  // false
}

// Enable gradient
torch::autograd::GradMode::set_enabled(true);
torch::autograd::GradMode::is_enabled();  // true
```

### Custom Autograd Function

```cpp
#include <torch/torch.h>

// Custom autograd function for a linear layer: y = x * W + b
struct MyLinearFunction : public torch::autograd::Function<MyLinearFunction> {
    // Forward: computes output and saves tensors needed for backward
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias) {

        ctx->save_for_backward({input, weight});
        auto output = input.mm(weight.t());
        if (bias.defined()) {
            output = output + bias;
        }
        return output;
    }

    // Backward: computes gradients
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {

        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];

        auto grad_output = grad_outputs[0];

        auto grad_input = grad_output.mm(weight);
        auto grad_weight = grad_output.t().mm(input);
        auto grad_bias = grad_output.sum(0);

        return {grad_input, grad_weight, grad_bias};
    }
};

// Usage
torch::Tensor input = torch::randn({8, 16}, torch::requires_grad());
torch::Tensor weight = torch::randn({32, 16}, torch::requires_grad());
torch::Tensor bias = torch::randn({32}, torch::requires_grad());

auto output = MyLinearFunction::apply(input, weight, bias);
auto loss = output.sum();
loss.backward();

auto grad_input = input.grad();
auto grad_weight = weight.grad();
auto grad_bias = bias.grad();
```

---

## Neural Network Modules

### torch::nn::Module Base Class

```cpp
#include <torch/torch.h>

// All neural network modules inherit from torch::nn::Module
class MyModule : public torch::nn::Module {
public:
    MyModule(int64_t in_features, int64_t out_features) {
        // Register parameters (will be tracked for gradients and serialization)
        W = register_parameter("W", torch::randn({out_features, in_features}));
        b = register_parameter("b", torch::randn({out_features}));

        // Register buffers (not parameters, but part of module state)
        running_mean = register_buffer("running_mean", torch::zeros({out_features}));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        return x.mm(W.t()) + b;
    }

    // Parameters and buffers are accessible
    torch::Tensor W;
    torch::Tensor b;
    torch::Tensor running_mean;
};

// Usage
auto model = MyModule(16, 32);
auto output = model.forward(torch::randn({8, 16}));

// Access named parameters
for (const auto& pair : model.named_parameters()) {
    std::cout << pair.key() << ": " << pair.value().sizes() << std::endl;
}

// Access all parameters as a vector
auto params = model.parameters();

// Move to CUDA
model->to(torch::kCUDA);

// Set to train/eval mode
model->train();
model->eval();
```

---

## Built-in Layers

### Linear

```cpp
#include <torch/torch.h>

// torch::nn::Linear
torch::nn::Linear linear(torch::nn::LinearOptions(16, 32)
    .bias(true)                     // include bias
    .dtype(torch::kFloat32));

// Forward
auto input = torch::randn({8, 16});
auto output = linear->forward(input);  // {8, 32}

// Without bias
torch::nn::Linear linear_no_bias(torch::nn::LinearOptions(16, 32).bias(false));
```

### Conv2d

```cpp
// torch::nn::Conv2d
torch::nn::Conv2d conv(torch::nn::Conv2dOptions(3, 16, /*kernel_size=*/3)
    .stride(1)
    .padding(1)
    .dilation(1)
    .groups(1)
    .bias(true)
    .padding_mode(torch::kZeros));

auto input = torch::randn({8, 3, 32, 32});
auto output = conv->forward(input);  // {8, 16, 32, 32}
```

### ConvTranspose2d

```cpp
torch::nn::ConvTranspose2d conv_t(
    torch::nn::ConvTranspose2dOptions(16, 3, 3)
        .stride(2)
        .padding(1)
        .output_padding(1));

auto input = torch::randn({8, 16, 16, 16});
auto output = conv_t->forward(input);  // {8, 3, 32, 32}
```

### LSTM

```cpp
// torch::nn::LSTM
torch::nn::LSTM lstm(torch::nn::LSTMOptions(64, 128)
    .num_layers(2)
    .batch_first(true)
    .dropout(0.1)
    .bidirectional(false));

auto input = torch::randn({8, 20, 64});  // batch, seq_len, input_size
auto output = lstm->forward(input);

// output.output:  {8, 20, 128}  (all hidden states)
// output.state.h: {2, 8, 128}   (final hidden state)
// output.state.c: {2, 8, 128}   (final cell state)
```

### GRU

```cpp
torch::nn::GRU gru(torch::nn::GRUOptions(64, 128)
    .num_layers(2)
    .batch_first(true)
    .dropout(0.1));

auto input = torch::randn({8, 20, 64});
auto output = gru->forward(input);
// output.output: {8, 20, 128}
// output.state:  {2, 8, 128}
```

### Transformer

```cpp
// torch::nn::Transformer
torch::nn::Transformer transformer(
    torch::nn::TransformerOptions()
        .d_model(512)
        .nhead(8)
        .num_encoder_layers(6)
        .num_decoder_layers(6)
        .dim_feedforward(2048)
        .dropout(0.1)
        .activation(torch::kReLU));

auto src = torch::randn({20, 8, 512});   // seq_len, batch, d_model
auto tgt = torch::randn({10, 8, 512});
auto out = transformer->forward(src, tgt); // {10, 8, 512}
```

### TransformerEncoder

```cpp
// Build a transformer encoder layer
auto encoder_layer = torch::nn::TransformerEncoderLayer(
    torch::nn::TransformerEncoderLayerOptions(512, 8)
        .dim_feedforward(2048)
        .dropout(0.1)
        .activation(torch::kReLU));

auto encoder = torch::nn::TransformerEncoder(
    torch::nn::TransformerEncoderOptions(encoder_layer, 6));

auto src = torch::randn({20, 8, 512});
auto out = encoder->forward(src);  // {20, 8, 512}
```

### BatchNorm

```cpp
// BatchNorm1d
torch::nn::BatchNorm1d bn1d(torch::nn::BatchNorm1dOptions(128)
    .eps(1e-5)
    .momentum(0.1)
    .affine(true)
    .track_running_stats(true));

// BatchNorm2d
torch::nn::BatchNorm2d bn2d(torch::nn::BatchNorm2dOptions(16)
    .eps(1e-5)
    .momentum(0.1));

auto input = torch::randn({8, 16, 32, 32});
auto output = bn2d->forward(input);
```

### Dropout

```cpp
// Dropout
torch::nn::Dropout dropout(torch::nn::DropoutOptions().p(0.5));

// Dropout2d (drops entire channels)
torch::nn::Dropout2d dropout2d(torch::nn::Dropout2dOptions().p(0.5));

// Alpha dropout (maintains self-normalizing properties)
torch::nn::AlphaDropout alpha_dropout(torch::nn::AlphaDropoutOptions().p(0.5));
```

### Embedding

```cpp
torch::nn::Embedding embedding(torch::nn::EmbeddingOptions(10000, 300)
    .padding_idx(0)
    .max_norm(1.0)
    .norm_type(2.0)
    .scale_grad_by_freq(false)
    .sparse(false));

auto indices = torch::randint(0, 10000, {8, 20});
auto embedded = embedding->forward(indices);  // {8, 20, 300}
```

### Pooling Layers

```cpp
// MaxPool2d
torch::nn::MaxPool2d maxpool(torch::nn::MaxPool2dOptions({2, 2})
    .stride({2, 2})
    .padding({0, 0}));

// AvgPool2d
torch::nn::AvgPool2d avgpool(torch::nn::AvgPool2dOptions({2, 2})
    .stride({2, 2}));

// AdaptiveAvgPool2d
torch::nn::AdaptiveAvgPool2d adap_pool(
    torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
```

### Activation Wrappers

```cpp
// Functional activations
auto relu_out = torch::relu(input);
auto sigmoid_out = torch::sigmoid(input);
auto tanh_out = torch::tanh(input);
auto gelu_out = torch::gelu(input);
auto leaky_out = torch::leaky_relu(input, 0.1);
auto softmax_out = torch::softmax(input, /*dim=*/1);
auto log_softmax_out = torch::log_softmax(input, /*dim=*/1);

// Module wrappers
torch::nn::ReLU relu_module;
torch::nn::GELU gelu_module;
torch::nn::Sigmoid sigmoid_module;
torch::nn::Tanh tanh_module;
torch::nn::SiLU silu_module;         // Swish activation
torch::nn::Mish mish_module;
torch::nn::LeakyReLU leaky_module(torch::nn::LeakyReLUOptions().negative_slope(0.01));
torch::nn::PReLU prelu_module;
```

---

## Loss Functions

```cpp
// CrossEntropyLoss (combines LogSoftmax + NLLLoss)
torch::nn::CrossEntropyLoss cross_entropy(
    torch::nn::CrossEntropyLossOptions()
        .weight(torch::tensor({1.0, 2.0, 1.0}))  // class weights
        .ignore_index(-100)
        .reduction(torch::kMean)
        .label_smoothing(0.0));

auto logits = torch::randn({8, 10});
auto targets = torch::randint(0, 10, {8});
auto loss = cross_entropy->forward(logits, targets);

// MSELoss
torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions(torch::kMean));
auto pred = torch::randn({8, 3});
auto target = torch::randn({8, 3});
auto mse = mse_loss->forward(pred, target);

// L1Loss
torch::nn::L1Loss l1_loss;

// BCELoss (Binary Cross Entropy, expects probabilities)
torch::nn::BCELoss bce_loss;

// BCEWithLogitsLoss (combines Sigmoid + BCELoss, more numerically stable)
torch::nn::BCEWithLogitsLoss bce_logits_loss(
    torch::nn::BCEWithLogitsLossOptions()
        .pos_weight(torch::tensor({2.0}))
        .reduction(torch::kMean));

// SmoothL1Loss (Huber loss)
torch::nn::SmoothL1Loss smooth_l1(torch::nn::SmoothL1LossOptions().beta(1.0));

// HuberLoss
torch::nn::HuberLoss huber_loss(torch::nn::HuberLossOptions().delta(1.0));

// KLDivLoss
torch::nn::KLDivLoss kl_div(torch::nn::KLDivLossOptions().reduction(torch::kBatchMean));

// CosineEmbeddingLoss
torch::nn::CosineEmbeddingLoss cosine_loss;

// TripletMarginLoss
torch::nn::TripletMarginLoss triplet_loss(
    torch::nn::TripletMarginLossOptions().margin(1.0).p(2));

// NLLLoss (Negative Log Likelihood, expects log-probabilities)
torch::nn::NLLLoss nll_loss;
```

---

## Optimizers

### SGD

```cpp
#include <torch/torch.h>

auto model = std::make_shared<MyModule>(16, 32);

// SGD optimizer
torch::optim::SGD optimizer(
    model->parameters(),
    torch::optim::SGDOptions(0.01)     // learning rate
        .momentum(0.9)
        .weight_decay(1e-4)
        .dampening(0)
        .nesterov(false));

// Training step
optimizer.zero_grad();
auto output = model->forward(input);
auto loss = cross_entropy->forward(output, target);
loss.backward();
optimizer.step();

// Learning rate adjustment
for (auto& group : optimizer.param_groups()) {
    auto& options = static_cast<torch::optim::SGDOptions&>(group.options());
    options.lr(options.lr() * 0.1);
}
```

### Adam

```cpp
torch::optim::Adam optimizer(
    model->parameters(),
    torch::optim::AdamOptions(0.001)     // learning rate
        .betas({0.9, 0.999})
        .eps(1e-8)
        .weight_decay(0)
        .amsgrad(false));

// Step
optimizer.zero_grad();
auto loss = compute_loss(model);
loss.backward();
optimizer.step();
```

### AdamW

```cpp
torch::optim::AdamW optimizer(
    model->parameters(),
    torch::optim::AdamWOptions(0.001)
        .betas({0.9, 0.999})
        .eps(1e-8)
        .weight_decay(0.01)
        .amsgrad(false));
```

### RMSprop

```cpp
torch::optim::RMSprop optimizer(
    model->parameters(),
    torch::optim::RMSpropOptions(0.01)
        .alpha(0.99)
        .eps(1e-8)
        .weight_decay(0)
        .momentum(0)
        .centered(false));
```

### Learning Rate Schedulers

```cpp
// Note: LibTorch C++ does not have built-in schedulers like Python
// You need to manually adjust learning rates

// Example: Step LR
auto adjust_lr = [&](int epoch, double initial_lr, int step_size, double gamma) {
    double lr = initial_lr * std::pow(gamma, epoch / step_size);
    for (auto& group : optimizer.param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
    }
};

// Cosine annealing
auto cosine_lr = [&](int epoch, int total_epochs, double initial_lr, double min_lr) {
    double lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(M_PI * epoch / total_epochs));
    for (auto& group : optimizer.param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
    }
};
```

---

## Data Loading

### Dataset

```cpp
#include <torch/torch.h>

// Custom dataset
class CustomDataset : public torch::data::Dataset<CustomDataset> {
public:
    // Example type returned by the dataset
    using ExampleType = torch::data::Example<>;

    CustomDataset(const std::string& data_path) {
        // Load data from files
        data_ = torch::randn({1000, 16});
        targets_ = torch::randint(0, 10, {1000});
    }

    // Return a single example
    ExampleType get(size_t index) override {
        return {data_[index], targets_[index]};
    }

    // Return the size of the dataset
    torch::optional<size_t> size() const override {
        return data_.size(0);
    }

private:
    torch::Tensor data_;
    torch::Tensor targets_;
};
```

### DataLoader

```cpp
// Create dataset
auto dataset = CustomDataset("/path/to/data")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

// Create data loader
auto data_loader = torch::data::make_data_loader<std::move>(dataset,
    torch::data::DataLoaderOptions()
        .batch_size(32)
        .workers(4)           // number of worker threads
        .drop_last(false));

// Iterate over batches
for (auto& batch : *data_loader) {
    auto data = batch.data;       // {32, 16}
    auto targets = batch.target;  // {32}
    // ... train
}
```

### DataLoaderOptions

```cpp
torch::data::DataLoaderOptions options;
options.batch_size(64);
options.workers(8);               // parallel data loading
options.drop_last(true);          // drop incomplete last batch
options.max_jobs(16);             // max queued batches
options.enforce_ordering(true);   // preserve order
options.channels_last(true);      // for images
```

---

## Serialization

### Saving and Loading Models

```cpp
#include <torch/torch.h>
#include <torch/serialize.h>

// === Save/Load Parameters ===

// Save
torch::save(model, "model.pt");

// Load
torch::load(model, "model.pt");

// === Save/Load Individual Tensors ===

torch::Tensor tensor = torch::randn({3, 4});
torch::save(tensor, "tensor.pt");

torch::Tensor loaded;
torch::load(loaded, "tensor.pt");

// === Save/Load Optimizer State ===

torch::save(optimizer, "optimizer.pt");
torch::load(optimizer, "optimizer.pt");

// === Save/Load state_dict (Python-compatible) ===

// Save
{
    torch::serialize::OutputArchive archive;
    model.save(archive);
    archive.save_to("model_state_dict.pt");
}

// Load
{
    torch::serialize::InputArchive archive;
    archive.load_from("model_state_dict.pt");
    model.load(archive);
}
```

### Full Checkpoint

```cpp
void save_checkpoint(std::shared_ptr<MyModule> model,
                     torch::optim::Optimizer& optimizer,
                     int epoch,
                     const std::string& path) {
    torch::serialize::OutputArchive archive;
    archive.write("epoch", epoch);
    model->save(archive);
    optimizer.save(archive);
    archive.save_to(path);
}

void load_checkpoint(std::shared_ptr<MyModule> model,
                     torch::optim::Optimizer& optimizer,
                     int& epoch,
                     const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    archive.read("epoch", epoch);
    model->load(archive);
    optimizer.load(archive);
}
```

---

## Custom Module Example (Complete)

```cpp
#include <torch/torch.h>
#include <iostream>

// A simple convolutional neural network
struct CNNImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::Dropout dropout{nullptr};

    CNNImpl(int64_t num_classes = 10)
        : conv1(torch::nn::Conv2dOptions(1, 32, 3).padding(1)),
          conv2(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
          bn1(32), bn2(64),
          fc(64 * 7 * 7, num_classes),
          dropout(torch::nn::DropoutOptions(0.25)) {

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("bn1", bn1);
        register_module("bn2", bn2);
        register_module("fc", fc);
        register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor x) {
        // Conv block 1
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 2);

        // Conv block 2
        x = conv2->forward(x);
        x = bn2->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 2);

        // Flatten and classify
        x = x.view({x.size(0), -1});
        x = dropout->forward(x);
        x = fc->forward(x);

        return x;
    }
};

// Use TORCH_MODULE macro for shared_ptr management
TORCH_MODULE(CNN);
```

---

## JIT Integration

### Loading TorchScript Models

```cpp
#include <torch/script.h>

// Load a TorchScript model (exported from Python)
torch::jit::script::Module module;
try {
    module = torch::jit::load("traced_model.pt");
} catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
}

// Move to CUDA
module.to(torch::kCUDA);

// Set to eval mode
module.eval();

// Run inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({1, 3, 224, 224}).to(torch::kCUDA));

torch::Tensor output = module.forward(inputs).toTensor();
std::cout << "Output shape: " << output.sizes() << std::endl;
```

### Saving TorchScript from C++

```cpp
// Export model to TorchScript via tracing
auto model = std::make_shared<CNNImpl>();
model->eval();

// Create example input
auto example_input = torch::randn({1, 1, 28, 28});

// Trace the model
torch::jit::script::Module traced_module = torch::jit::trace(
    std::shared_ptr<CNNImpl>(model),
    example_input);

// Save
traced_module.save("traced_cnn.pt");
```

### JIT Module Methods

```cpp
torch::jit::script::Module module = torch::jit::load("model.pt");

// Get method names
for (const auto& method : module.get_methods()) {
    std::cout << "Method: " << method.name() << std::endl;
}

// Access attributes
for (const auto& attr : module.named_attributes()) {
    std::cout << "Attr: " << attr.name << std::endl;
}

// Access parameters
for (const auto& param : module.named_parameters()) {
    std::cout << "Param: " << param.name
              << " shape: " << param.value.sizes() << std::endl;
}

// Run a specific method
auto result = module.run_method("forward", input_tensor);
```

---

## CUDA Integration

### Device Management

```cpp
// Check CUDA availability
bool cuda_available = torch::cuda::is_available();
int device_count = torch::cuda::device_count();

// Set device
torch::cuda::manual_seed(42);
torch::cuda::manual_seed_all(42);  // all devices

// Get device properties
torch::cuda::DeviceProp props = torch::cuda::getDeviceProperties(0);
std::cout << "Device: " << props.name << std::endl;
std::cout << "Memory: " << props.total_memory << std::endl;

// Synchronize
torch::cuda::synchronize();
torch::cuda::synchronize(0);  // specific device

// Current device
int current = torch::cuda::current_device();
```

### Memory Management

```cpp
// Memory statistics
torch::cuda::DeviceStats stats = torch::cuda::getDeviceStats(0);

// Reset peak memory stats
torch::cuda::resetPeakMemoryStats();
torch::cuda::resetPeakMemoryStats(0);

// Empty cache
torch::cuda::empty_cache();

// Memory summary
auto allocated = torch::cuda::memory_stats();
// Keys: "allocated_bytes.all.current", "allocated_bytes.all.peak", etc.
```

### Multi-GPU

```cpp
// Move tensor to specific GPU
auto t0 = torch::randn({3, 4}).to(torch::Device(torch::kCUDA, 0));
auto t1 = torch::randn({3, 4}).to(torch::Device(torch::kCUDA, 1));

// DataParallel (limited C++ support)
// Recommended: use TorchScript with device placement
```

---

## Complete Training Loop Example

```cpp
#include <torch/torch.h>
#include <iostream>

// Dataset for MNIST-like data
struct RandomDataset : torch::data::Dataset<RandomDataset> {
    torch::Tensor data_, labels_;

    RandomDataset(int64_t n_samples, int64_t n_features, int64_t n_classes) {
        data_ = torch::randn({n_samples, n_features});
        labels_ = torch::randint(0, n_classes, {n_samples});
    }

    torch::data::Example<> get(size_t index) override {
        return {data_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return data_.size(0);
    }
};

// Simple neural network
struct NetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    NetImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim)
        : fc1(input_dim, hidden_dim),
          fc2(hidden_dim, hidden_dim),
          fc3(hidden_dim, output_dim) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(Net);

int main() {
    // Hyperparameters
    const int64_t input_dim = 784;
    const int64_t hidden_dim = 256;
    const int64_t output_dim = 10;
    const int64_t batch_size = 64;
    const double learning_rate = 0.001;
    const int64_t num_epochs = 10;

    // Device
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Training on: " << device << std::endl;

    // Model
    Net model(input_dim, hidden_dim, output_dim);
    model->to(device);

    // Loss and optimizer
    torch::nn::CrossEntropyLoss criterion;
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(learning_rate));

    // Data
    auto dataset = RandomDataset(1000, input_dim, output_dim)
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<std::move>(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size));

    // Training loop
    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;
        int64_t correct = 0;
        int64_t total = 0;

        model->train();

        for (auto& batch : *data_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            // Forward pass
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = criterion->forward(output, targets);

            // Backward pass
            loss.backward();
            optimizer.step();

            // Statistics
            running_loss += loss.item<double>() * data.size(0);
            auto predictions = output.argmax(1);
            correct += predictions.eq(targets).sum().item<int64_t>();
            total += data.size(0);
        }

        double avg_loss = running_loss / total;
        double accuracy = static_cast<double>(correct) / total * 100;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                  << "Loss: " << avg_loss << " "
                  << "Accuracy: " << accuracy << "%" << std::endl;
    }

    // Save model
    torch::save(model, "model.pt");
    std::cout << "Model saved to model.pt" << std::endl;

    return 0;
}
```

---

## Summary

LibTorch provides a complete C++ API for PyTorch:

1. **Tensors**: Full tensor creation, manipulation, and computation via ATen
2. **Autograd**: Custom functions via `torch::autograd::Function`, gradient computation
3. **NN Modules**: `torch::nn::Module` with `register_parameter`, `forward`, and built-in layers
4. **Layers**: Linear, Conv2d, LSTM, Transformer, BatchNorm, Dropout, Embedding, and more
5. **Losses**: CrossEntropyLoss, MSELoss, L1Loss, BCELoss, and others
6. **Optimizers**: SGD, Adam, AdamW, RMSprop with configurable options
7. **Data Loading**: `torch::data::Dataset`, `DataLoader`, transforms
8. **Serialization**: `torch::save`, `torch::load`, checkpoint management
9. **JIT**: Load and run TorchScript models, tracing from C++
10. **CUDA**: Device management, memory tracking, multi-GPU support
