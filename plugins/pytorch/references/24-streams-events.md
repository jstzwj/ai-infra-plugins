# CUDA Streams, Events, and Graphs

CUDA concurrency primitives for overlapping computation, communication, and data transfers.

```python
import torch
```

---

## CUDA Streams

Streams are independent queues of GPU operations. Operations in different streams can execute concurrently.

### Stream Constructor

```python
stream = torch.cuda.Stream(
    device: Union[int, torch.device] = None,
    priority: int = 0,              # lower number = higher priority
)
```

### Stream Methods and Properties

```python
stream.device           # torch.device for this stream
stream.priority         # int priority
stream.query()          # True if all ops completed
stream.synchronize()    # block until all ops complete
stream.record_event(event=None)  # record an Event on this stream
stream.wait_event(event)         # wait for event before continuing
stream.wait_stream(other_stream) # wait for other stream to complete
```

### Current Stream Management

```python
torch.cuda.current_stream(device=None) -> Stream
torch.cuda.set_stream(stream)            # set active stream
torch.cuda.default_stream(device=None)   # default (priority 0) stream
```

### Stream Context Manager

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    # Operations go on s1
    a = torch.randn(1000, device="cuda")
    b = a @ a.T

with torch.cuda.stream(s2):
    # Concurrent with s1
    c = torch.randn(1000, device="cuda")
    d = c @ c.T

torch.cuda.synchronize()  # wait for both
```

### Multi-Stream Data Loading

```python
preload_stream = torch.cuda.Stream()

for batch_idx, (data, target) in enumerate(dataloader):
    # Wait for previous preload to finish
    torch.cuda.current_stream().wait_stream(preload_stream)
    data = data_on_gpu
    target = target_on_gpu

    # Start preloading next batch on separate stream
    with torch.cuda.stream(preload_stream):
        next_data, next_target = next(dataloader_iter)
        next_data = next_data.cuda(non_blocking=True)
        next_target = next_target.cuda(non_blocking=True)

    # Compute on current batch while next batch loads
    output = model(data)
    loss = criterion(output, target)
```

---

## CUDA Events

Events are synchronization markers between streams. Used for timing and dependencies.

### Event Constructor

```python
event = torch.cuda.Event(
    enable_timing: bool = False,
    blocking: bool = False,
    interprocess: bool = False,
)
```

### Event Methods

```python
event.record(stream=None)               # record event on stream
event.wait(stream=None)                 # wait for this event on stream
event.synchronize()                     # block CPU until event completes
event.query()                           # True if event completed
event.elapsed_time(end_event)           # milliseconds between two events
```

### Timing with Events

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(inputs)
loss = criterion(output, target)
loss.backward()
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
print(f"Forward+Backward: {elapsed_ms:.2f} ms")
```

### Inter-Stream Synchronization

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    result1 = heavy_compute_1()
    event1 = s1.record_event()

with torch.cuda.stream(s2):
    s2.wait_event(event1)  # s2 waits for s1 to finish
    result2 = heavy_compute_2(result1)
```

---

## CUDA Graphs

CUDA Graphs capture a series of GPU operations into a single graph for efficient replay, eliminating CPU launch overhead.

### CUDAGraph

```python
g = torch.cuda.CUDAGraph()

# Warmup (required before capture)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        output = model(static_input)

torch.cuda.current_stream().wait_stream(s)

# Capture
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (very fast, no CPU overhead)
g.replay()
```

### Static Shapes and Inputs

CUDA Graphs require static shapes. Inputs must be pre-allocated.

```python
# Pre-allocate static buffers
static_input = torch.randn(batch_size, seq_len, d_model, device="cuda")
static_target = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

g = torch.cuda.CUDAGraph()

# Warmup with loss
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = model(static_input)
        static_loss = loss_fn(static_output, static_target)
        static_loss.backward()
torch.cuda.current_stream().wait_stream(s)

# Capture full forward + backward
with torch.cuda.graph(g):
    static_output = model(static_input)
    static_loss = loss_fn(static_output, static_target)
    static_loss.backward()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Copy new data into static buffers
        static_input.copy_(batch[0])
        static_target.copy_(batch[1])
        optimizer.zero_grad()
        g.replay()
        optimizer.step()
```

### Multiple Graphs

```python
# Separate graphs for different batch sizes
graphs = {}
for bs in [8, 16, 32]:
    g = torch.cuda.CUDAGraph()
    inp = torch.randn(bs, features, device="cuda")
    # warmup + capture ...
    graphs[bs] = (g, inp)

# Replay appropriate graph
g, inp = graphs[current_batch_size]
inp.copy_(data)
g.replay()
```

### Graph Memory Pooling

```python
# Use private memory pool for graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, pool=other_graph.pool()):
    # Shares memory pool with other graph
    output = model(input)
```

---

## Non-Blocking Data Transfer

```python
# Asynchronous H2D transfer on separate stream
transfer_stream = torch.cuda.Stream()
with torch.cuda.stream(transfer_stream):
    gpu_data = cpu_data.cuda(non_blocking=True)

# Overlap transfer with compute
compute_stream = torch.cuda.Stream()
with torch.cuda.stream(compute_stream):
    compute_stream.wait_stream(transfer_stream)
    result = model(gpu_data)
```

---

## Pinning Memory for Faster Transfer

```python
# Pin CPU memory for faster async transfer to GPU
cpu_tensor = torch.randn(1000, 1000).pin_memory()
# Now .cuda(non_blocking=True) is faster
gpu_tensor = cpu_tensor.cuda(non_blocking=True)
```

---

## Complete Multi-Stream Training Example

```python
import torch

preload_stream = torch.cuda.Stream()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters())

static_input = torch.randn(32, 784, device="cuda")
static_loss = torch.tensor(0.0, device="cuda")

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = model(static_input)
        l = loss_fn(out)
        l.backward()
torch.cuda.current_stream().wait_stream(s)

with torch.cuda.graph(g):
    static_output = model(static_input)
    static_loss = loss_fn(static_output)
    static_loss.backward()

for data, target in dataloader:
    static_input.copy_(data.cuda())
    optimizer.zero_grad()
    g.replay()
    optimizer.step()
```
