# Autograd Engine (C++ Implementation)

## Overview

PyTorch's autograd engine is responsible for computing gradients during the backward pass. While the Python API provides `tensor.backward()` and `torch.autograd.grad()`, the actual gradient computation happens in the C++ engine located at `torch/csrc/autograd/`. The engine builds a directed acyclic graph (DAG) during the forward pass and traverses it in topological order during backward.

**Source location**: `torch/csrc/autograd/engine.cpp`, `torch/csrc/autograd/`

---

## Core Engine Components

### EvalState

`EvalState` tracks whether a particular graph task has already been evaluated (completed backward). This prevents re-execution of already-completed backward passes.

```cpp
// torch/csrc/autograd/eval_state.h

struct EvalState {
    // Track which nodes have been evaluated
    // Prevents duplicate evaluation in the backward graph
};
```

### ReadyQueue

`ReadyQueue` is a thread-safe priority queue that holds `NodeTask` objects waiting to be executed. Each device has its own `ReadyQueue` to enable parallel gradient computation across devices.

```cpp
// torch/csrc/autograd/ready_queue.h

class ReadyQueue {
public:
    void push(NodeTask item, bool incrementOutstandingTasks = true);
    NodeTask pop();
    bool empty() const;
    size_t size() const;

private:
    std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime>
        heap_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
};
```

#### Queue Design

- One `ReadyQueue` per device (CPU has one, each CUDA device has one)
- Priority is based on the topological order of the computation graph
- Thread-safe: multiple worker threads can push and pop simultaneously
- Blocks on `pop()` when queue is empty (waiting for new tasks)

### NodeTask

`NodeTask` represents a single unit of work for the autograd engine: computing the gradient for one node.

```cpp
// torch/csrc/autograd/task.h

struct NodeTask {
    // The graph task this node belongs to
    std::shared_ptr<GraphTask> base_;

    // The autograd node (function) to execute
    std::shared_ptr<Node> fn_;

    // Input gradients (from downstream)
    variable_list grads_;

    // The device this task should run on
    c10::Device device_;

    // Whether this is a root task (triggered by backward() call)
    bool is_root_task_ = false;

    // Timestamp for priority ordering
    int64_t getRecomputeDepth() const;
};
```

---

## GraphTask

`GraphTask` captures the entire backward graph for a single `backward()` call. It manages the state of the backward pass including outstanding tasks, captured variables, and error handling.

```cpp
// torch/csrc/autograd/graph_task.h

struct GraphTask {
    // =========================================
    // Graph Structure
    // =========================================

    // Root nodes (final outputs) of the backward graph
    std::vector<Node*> exec_info_;

    // All nodes in the graph
    // Used for topological ordering

    // =========================================
    // Execution State
    // =========================================

    // Number of outstanding (not yet completed) tasks
    std::atomic<uint64_t> outstanding_tasks_{0};

    // Whether the backward has been completed
    std::atomic_bool completed_{false};

    // =========================================
    // Error Handling
    // =========================================

    // First exception encountered during backward
    std::exception_ptr exception_;

    // Whether to detect anomalies (for debugging)
    bool debug_mode_ = false;

    // =========================================
    // Gradient Accumulation
    // =========================================

    // Map from variable ID to accumulated gradient
    std::unordered_map<uint64_t, Variable> accumulated_gradients_;

    // =========================================
    // Hooks
    // =========================================

    // Pre/post hooks for gradient computation
    std::vector<std::function<void()>> pre_hooks_;
    std::vector<std::function<void()>> post_hooks_;

    // =========================================
    // Configuration
    // =========================================

    // Whether to keep the graph after backward
    bool keep_graph_ = false;

    // Whether to compute gradients for leaf variables only
    bool grad_mode_ = true;

    // The device the backward was initiated on
    c10::Device device_;

    // =========================================
    // Threading
    // =========================================

    // Ready queues for each device
    std::vector<std::shared_ptr<ReadyQueue>> ready_queues_;

    // The main thread's ready queue (for final results)
    std::shared_ptr<ReadyQueue> owner_ready_queue_;
};
```

---

## How backward() Executes

### Execution Flow

```
1. torch.autograd.backward(tensors, grad_tensors)
    |
    v
2. C++ Engine::execute(root_nodes, grads, keep_graph, create_graph)
    |
    v
3. Create GraphTask
    - Capture the backward graph
    - Set up ready queues
    |
    v
4. Topological sort
    - Order nodes from outputs to inputs
    - Compute dependencies (number of inputs each node needs)
    |
    v
5. Push root tasks to ReadyQueue
    - One task per output tensor
    |
    v
6. Worker threads process tasks:
    a. Pop task from ReadyQueue
    b. Call Node::apply(grads) to compute gradients
    c. Accumulate gradients for inputs
    d. Decrement dependency counts for parent nodes
    e. When all dependencies met, push parent to ReadyQueue
    |
    v
7. Wait for outstanding_tasks_ to reach 0
    |
    v
8. Return accumulated gradients
```

### Engine::execute (Simplified)

```cpp
// torch/csrc/autograd/engine.cpp

variable_list Engine::execute(
    const edge_list& root_edges,     // output edges
    const variable_list& grads,       // initial gradients
    bool keep_graph,                  // retain graph after backward
    bool create_graph,                // create new graph for higher-order grads
    const edge_list& output_edges     // specific outputs to compute
) {
    // 1. Validate inputs
    TORCH_CHECK(root_edges.size() == grads.size(), "Mismatched sizes");

    // 2. Create GraphTask
    auto graph_task = std::make_shared<GraphTask>(
        keep_graph, create_graph, /* ... */);

    // 3. Compute dependencies (topological sort)
    //    For each node, count how many downstream nodes feed into it
    compute_dependencies(graph_task, root_edges);

    // 4. Set up device-ready queues
    //    Each device gets its own queue for parallel execution
    initialize_device_queues(graph_task);

    // 5. Push root tasks
    for (auto i = 0; i < root_edges.size(); ++i) {
        auto& edge = root_edges[i];
        auto& grad = grads[i];
        auto task = NodeTask(graph_task, edge.function, {grad}, edge.function->device());
        graph_task->ready_queues_[task.device_]->push(task);
    }

    // 6. Wait for completion
    auto result = graph_task->owner_ready_queue_->pop();

    // 7. Check for exceptions
    if (graph_task->exception_) {
        std::rethrow_exception(graph_task->exception_);
    }

    // 8. Return accumulated gradients
    return collect_gradients(graph_task, output_edges);
}
```

### Topological Sort and Dependency Computation

```cpp
void Engine::compute_dependencies(
    const std::shared_ptr<GraphTask>& graph_task,
    const edge_list& root_edges
) {
    // BFS from root edges backward through the graph
    // For each node, count the number of edges feeding into it
    // A node can only execute when all its inputs have been computed

    std::unordered_map<Node*, int> dependencies;

    // Start BFS from roots
    std::queue<Node*> queue;
    for (const auto& edge : root_edges) {
        queue.push(edge.function.get());
    }

    while (!queue.empty()) {
        auto* node = queue.front();
        queue.pop();

        for (const auto& input_edge : node->next_edges()) {
            auto* parent = input_edge.function.get();
            dependencies[parent]++;
            if (!visited[parent]) {
                visited[parent] = true;
                queue.push(parent);
            }
        }
    }
}
```

### Task Execution

```cpp
void Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) {
    while (true) {
        // Get next task from the ready queue
        auto task = ready_queue()->pop();

        if (task.is_shutdown_) {
            break;
        }

        // Execute the node
        try {
            execute_task(task);
        } catch (const std::exception& e) {
            // Store exception and mark as completed
            graph_task->exception_ = std::current_exception();
            graph_task->completed_ = true;
            break;
        }
    }
}

void Engine::execute_task(const NodeTask& task) {
    auto* fn = task.fn_.get();
    auto& grads = task.grads_;

    // Call the node's apply function
    auto output_grads = fn->apply(grads);

    // Propagate gradients to parent nodes
    auto& next_edges = fn->next_edges();
    for (auto i = 0; i < next_edges.size(); ++i) {
        auto& edge = next_edges[i];
        if (output_grads[i].defined()) {
            auto parent_task = NodeTask(
                task.base_, edge.function, {output_grads[i]},
                edge.function->device());

            // Check if all dependencies for parent are met
            if (--graph_task->dependencies_[edge.function.get()] == 0) {
                // All inputs ready, push to queue
                task.base_->ready_queues_[parent_task.device_]->push(parent_task);
            } else {
                // Accumulate gradient
                graph_task->accumulate_gradient(edge.function, output_grads[i]);
            }
        }
    }

    // Decrement outstanding tasks
    if (--task.base_->outstanding_tasks_ == 0) {
        task.base_->completed_ = true;
        task.base_->owner_ready_queue_->push(NodeTask(/* sentinel */));
    }
}
```

---

## Compiled Autograd: torch._compiled_autograd

PyTorch 2.0+ can compile the autograd engine itself using `torch._compiled_autograd`.

```python
import torch

# Enable compiled autograd
torch._compiled_autograd.enable()

# Now backward passes are compiled and optimized
x = torch.randn(10, requires_grad=True)
y = (x * 2).sum()
y.backward()  # backward is compiled

# Disable
torch._compiled_autograd.disable()
```

### How Compiled Autograd Works

1. During the first backward pass, the autograd graph is captured
2. The graph is compiled using torch.compile into optimized code
3. Subsequent backward passes with the same graph structure use the compiled version
4. Dynamic graphs get recompiled when the structure changes

### Benefits

- Eliminates Python overhead during backward
- Fuses gradient computations
- Optimizes memory access patterns
- Can be significantly faster for repeated backward passes

---

## torch.autograd.graph

### Node

The C++ `Node` class represents a function in the computation graph.

```cpp
// torch/csrc/autograd/function.h

class Node {
public:
    // Constructor
    explicit Node(uint64_t sequence_number, Edge::vertex_list next_edges = {});

    // Core method: compute backward
    virtual variable_list apply(variable_list&& grads) = 0;

    // Graph structure
    Edge::vertex_list next_edges() const;
    void set_next_edges(Edge::vertex_list&& edges);
    void add_next_edge(Edge edge);
    Edge next_edge(size_t index) const;

    // Metadata
    uint64_t sequence_number() const;
    std::string name() const;

    // Topology
    const std::vector<Snapshot>& saved_tensors() const;
    void save_for_backward(variable_list tensors);

    // Release saved tensors (after backward)
    void release_variables();

    // Device
    c10::Device device() const;

    // Tracing
    bool is_traceable() const;

protected:
    uint64_t sequence_number_;
    Edge::vertex_list next_edges_;
    variable_list saved_tensors_;
};
```

### saved_tensors_hooks

Save tensor hooks allow intercepting the tensors saved during the forward pass, enabling memory optimization and custom packing.

```python
import torch

class PackHook:
    def __init__(self, pack_fn):
        self.pack_fn = pack_fn

    def __call__(self, tensor):
        return self.pack_fn(tensor)

class UnpackHook:
    def __init__(self, unpack_fn):
        self.unpack_fn = unpack_fn

    def __call__(self, packed):
        return self.unpack_fn(packed)

# Example: save tensors in FP16 to reduce memory
def pack(tensor):
    return tensor.half()

def unpack(packed):
    return packed.float()

with torch.autograd.graph.saved_tensors_hooks(PackHook(pack), UnpackHook(unpack)):
    x = torch.randn(100, 100, requires_grad=True)
    y = x.mm(x)
    y.sum().backward()
    # During forward, tensors are saved in FP16
    # During backward, they are unpacked to FP32
```

---

## Gradient Accumulation

### How Gradients Accumulate

```cpp
// When backward computes a gradient for a tensor that already has a gradient,
// the new gradient is added (accumulated) rather than replaced.

// In the engine:
void accumulate_gradient(
    std::shared_ptr<GraphTask>& graph_task,
    Node* node,
    const Variable& grad
) {
    auto& accumulated = graph_task->accumulated_gradients_[node->tensor_id];
    if (accumulated.defined()) {
        accumulated = accumulated + grad;  // accumulate
    } else {
        accumulated = grad;  // first gradient
    }
}
```

### Python-Level Gradient Accumulation

```python
# Manual gradient accumulation
model.zero_grad()  # or optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # accumulates gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### set_to_none=True

```python
# More memory-efficient gradient zeroing
optimizer.zero_grad(set_to_none=True)
# Instead of filling gradients with zeros, sets them to None
# Saves memory and avoids unnecessary operations
```

---

## Higher-Order Gradients

The autograd engine supports computing gradients of gradients (second-order, third-order, etc.).

```python
import torch

# Second-order gradients
x = torch.randn(3, requires_grad=True)
y = x ** 3

# First backward creates a new computation graph
grad1 = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
# grad1 = 3 * x^2, which is itself differentiable

# Second backward
grad2 = torch.autograd.grad(grad1.sum(), x)[0]
# grad2 = 6 * x
```

### Engine Support for Higher-Order Gradients

```cpp
// When create_graph=True in backward():
// 1. The gradient computation itself builds a new computation graph
// 2. This new graph can be differentiated again
// 3. The engine re-enters itself recursively

variable_list Engine::execute(
    const edge_list& root_edges,
    const variable_list& grads,
    bool keep_graph,
    bool create_graph,  // <-- enables higher-order gradients
    const edge_list& output_edges
) {
    if (create_graph) {
        // Enable gradient tracking during backward
        // The gradient operations become differentiable
        torch::autograd::GradMode::enable(true);
    }
    // ... normal execution
}
```

---

## Thread Safety in Autograd

### Thread Safety Guarantees

1. **Multiple forward passes** from different threads are safe
2. **Multiple backward passes** from different threads are safe (each has its own GraphTask)
3. **A single backward pass** is parallelized across devices by the engine

### Thread-Local State

```cpp
// torch/csrc/autograd/autograd.h

// Thread-local gradient mode
class GradMode {
    static bool is_enabled();
    static void set_enabled(bool enabled);
    // Thread-local: each thread has its own setting
};

// Thread-local anomaly detection
class AnomalyMode {
    static bool is_enabled();
    static void set_enabled(bool enabled);
    static std::string stack_trace();
};
```

### Multithreaded Backward

```python
import torch
from torch import autograd

# Enable multithreading for backward
autograd.set_multithreading_enabled(True)

# The engine will use multiple threads to compute gradients
# Tasks for different devices are executed in parallel
```

```cpp
// torch/csrc/autograd/engine.cpp

// Number of worker threads in the autograd engine
// Default: number of devices (1 for CPU-only)
int Engine::num_workers() const {
    return std::max(1, static_cast<int>(ready_queues_.size()));
}
```

---

## Autograd Anomaly Detection

### C++ Implementation

```cpp
// When anomaly detection is enabled:
// 1. Each Node records the Python stack trace during forward
// 2. If backward encounters an error, the stack trace is printed
// 3. This helps identify which forward operation created the problematic node

class AnomalyMode {
    static thread_local bool _enabled;
    static thread_local std::string _stack_trace;

public:
    static bool is_enabled() { return _enabled; }
    static void set_enabled(bool enabled) { _enabled = enabled; }

    static void store_stack_trace() {
        if (_enabled) {
            _stack_trace = get_python_stack_trace();
        }
    }
};

// In Node constructor:
Node::Node(...) {
    if (AnomalyMode::is_enabled()) {
        anomaly_metadata_ = std::make_unique<AnomalyMetadata>();
        anomaly_metadata_->store_stack_trace();
    }
}
```

### Python Usage

```python
import torch

# Enable anomaly detection
with torch.autograd.detect_anomaly():
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    y.backward(torch.tensor([1, 0, float('nan')]))
    # If NaN is detected, prints the forward operation's stack trace
```

---

## Custom Autograd Function C++ Registration

### Custom Function in C++

```cpp
#include <torch/torch.h>

// Define a custom autograd function
struct MyReluFunction : public torch::autograd::Function<MyReluFunction> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input) {
        // Save mask for backward
        ctx->save_for_backward({input});
        return torch::relu(input);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        // ReLU backward: gradient * (input > 0)
        auto mask = (input > 0).to(grad_output.dtype());
        return {grad_output * mask};
    }
};

// Use in forward pass
torch::Tensor my_relu(torch::Tensor x) {
    return MyReluFunction::apply(x);
}
```

### Registering with the Dispatcher

```cpp
// Register as a custom operator with autograd support
TORCH_LIBRARY(myops, m) {
    m.def("custom_relu(Tensor self) -> Tensor");
}

// Meta kernel (shape inference)
TORCH_LIBRARY_IMPL(myops, Meta, m) {
    m.impl("custom_relu", [](const Tensor& self) {
        return torch::empty_like(self);
    });
}

// Autograd implementation using custom function
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
    m.impl("custom_relu", [](const Tensor& self) {
        return MyReluFunction::apply(self);
    });
}
```

---

## Memory Management During Backward

### Saved Tensor Management

```cpp
// During forward, tensors are saved for backward:
// - Input tensors that are needed for gradient computation
// - Output tensors (for operations like ReLU where the output determines the gradient)

// Memory management strategies:
// 1. Default: keep saved tensors in memory until backward completes
// 2. Hooks: intercept saved tensors for custom storage (e.g., offload to CPU)
// 3. Checkpointing: recompute saved tensors during backward instead of storing

// Saved tensor reference counting
class SavedVariable {
    void reset() {
        // Release the saved tensor
        variable_.reset();
    }

    Variable unpack() const {
        // Return the saved tensor
        // If tensor was packed via hooks, unpack it first
        return variable_;
    }
};
```

### Gradient Checkpointing

```python
# Memory-efficient backward using gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward(x):
    # Instead of:
    # y = expensive_op1(x)
    # z = expensive_op2(y)
    # return z

    # Use checkpointing:
    y = checkpoint(expensive_op1, x)
    z = checkpoint(expensive_op2, y)
    return z

# During backward:
# 1. y is recomputed from x (not stored)
# 2. z is recomputed from y (not stored)
# This trades computation for memory
```

### Gradient Checkpointing C++ Implementation

```cpp
// torch/csrc/autograd/functions/compat.h

struct CheckpointFunction : public Node {
    // During forward:
    // - Run the forward function without saving intermediates
    // - Only save the inputs

    // During backward:
    // - Recompute the forward pass
    // - Then compute gradients through the recomputed graph
    // - Discard the recomputed graph
};
```

---

## Saved Tensor Hooks: Pack/Unpack

### C++ Implementation

```cpp
// torch/csrc/autograd/saved_variable.h

class SavedVariable {
public:
    // Pack: called when saving a tensor during forward
    void pack(const Variable& tensor) {
        if (pack_hook_) {
            // Custom packing (e.g., compress, move to different device)
            packed_data_ = pack_hook_(tensor);
        } else {
            variable_ = tensor;
        }
    }

    // Unpack: called when accessing the tensor during backward
    Variable unpack() const {
        if (pack_hook_) {
            // Custom unpacking (e.g., decompress, move back)
            return unpack_hook_(packed_data_);
        }
        return variable_;
    }

private:
    Variable variable_;
    std::any packed_data_;
    std::function<std::any(const Variable&)> pack_hook_;
    std::function<Variable(const std::any&)> unpack_hook_;
};
```

---

## Engine Configuration

### Thread Configuration

```python
# torch/csrc/autograd/engine.h

class Engine {
public:
    // Number of worker threads
    // Default: number of CUDA devices + 1 (for CPU)
    virtual int num_workers() const;

    // Set number of worker threads
    // Only effective before first backward call
    static void set_num_threads(int threads);
};
```

### Environment Variables

```bash
# Maximum number of backward threads
export TORCH_NUM_BACKWARD_THREADS=4

# Disable multithreaded backward
export TORCH_DISABLE_MULTITHREADED_BACKWARD=1
```

---

## Summary

PyTorch's C++ autograd engine provides efficient gradient computation:

1. **GraphTask**: Captures the entire backward graph for a single backward() call
2. **ReadyQueue**: Thread-safe per-device queue for task scheduling
3. **NodeTask**: Single unit of work -- computing gradient for one node
4. **Topological sort**: Determines execution order and dependencies
5. **Parallel execution**: Tasks for different devices run in parallel
6. **Higher-order gradients**: Supported via create_graph=True
7. **Thread safety**: Multiple backward passes can run concurrently
8. **Saved tensor hooks**: Custom packing/unpacking for memory optimization
9. **Compiled autograd**: torch.compile can optimize the backward pass itself
10. **Anomaly detection**: Records forward stack traces for debugging backward errors
