# PyTorch - Chapter 52: Inference Optimization

This reference covers techniques for optimizing PyTorch model inference.

---

## 52.1 Inference Mode

```python
# Fastest: no autograd overhead
with torch.inference_mode():
    output = model(input)

# Slightly slower but more compatible
with torch.no_grad():
    output = model(input)

# Decorator
@torch.inference_mode()
def predict(model, input):
    return model(input)
```

---

## 52.2 JIT Optimization

```python
# Freeze: inline constants, fold computations
frozen = torch.jit.freeze(torch.jit.script(model))

# Optimize for inference
optimized = torch.jit.optimize_for_inference(frozen)
```

---

## 52.3 torch.compile for Inference

```python
# Compile with CUDA Graphs for lowest latency
model = torch.compile(model, mode="reduce-overhead")

# Full autotuning for maximum throughput
model = torch.compile(model, mode="max-autotune")
```

---

## 52.4 CUDA Graphs

```python
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    static_input = input.clone()
    static_output = model(static_input)

with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (very fast, no Python overhead)
static_input.copy_(new_input)
g.replay()
output = static_output.clone()
```

---

## 52.5 CPU Optimization

```python
# MKL-DNN / oneDNN
torch.backends.mkldnn.enabled = True

# Thread settings
torch.set_num_threads(4)             # Match core count
torch.set_num_interop_threads(2)     # Inter-op parallelism

# numactl for NUMA systems
# numactl --cpunodebind=0 --membind=0 python infer.py
```

---

## 52.6 Memory Optimization

```python
# Gradient checkpointing (trades compute for memory)
from torch.utils.checkpoint import checkpoint
output = checkpoint(model, input, use_reentrant=False)

# Model offloading
model = model.to('cpu')  # Offload to CPU when not in use

# BFloat16 for lower memory
model = model.to(torch.bfloat16)
```
