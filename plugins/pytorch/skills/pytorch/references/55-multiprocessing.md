# PyTorch - Chapter 55: Multiprocessing

This reference covers torch.multiprocessing for parallel computation.

---

## 55.1 torch.multiprocessing.spawn

```python
torch.multiprocessing.spawn(
    fn,                     # Function(worker_id, *args)
    args=(),                # Arguments to fn
    nprocs=1,               # Number of processes
    join=True,              # Wait for all to finish
    daemon=False,           # Daemon mode
    start_method='spawn',   # fork, spawn, or forkserver
)
```

```python
def train_worker(rank, world_size):
    # Setup distributed training on this worker
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # ... training code ...

torch.multiprocessing.spawn(train_worker, args=(4,), nprocs=4)
```

---

## 55.2 Start Methods

| Method | Notes |
|--------|-------|
| `fork` | Copy process state. NOT safe with CUDA |
| `spawn` | Fresh process. Required for CUDA sharing |
| `forkserver` | Server process. Safe but slower |

---

## 55.3 Sharing Tensors

```python
# Shared memory (CPU only)
t = torch.randn(10)
t.share_memory_()     # Move to shared memory
t.is_shared()         # True

# CUDA IPC (for multi-process GPU)
# Must use spawn or forkserver start method
```
