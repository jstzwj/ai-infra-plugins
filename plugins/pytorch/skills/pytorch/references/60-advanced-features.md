# PyTorch - Chapter 60: Advanced Features

This reference covers miscellaneous advanced features, utilities, and debugging tools.

---

## 60.1 DLPack Interop

```python
# PyTorch → JAX/TensorFlow/etc.
from torch.utils.dlpack import to_dlpack, from_dlpack
dlpack_tensor = to_dlpack(torch_tensor)

# Other framework → PyTorch
torch_tensor = from_dlpack(dlpack_tensor)
```

---

## 60.2 TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_histogram('weights', model.linear.weight, epoch)
writer.add_graph(model, input)
writer.add_image('input', img_grid, epoch)
writer.close()
```

---

## 60.3 Custom C++/CUDA Extensions

```python
from torch.utils.cpp_extension import CppExtension, CUDAExtension, load, load_inline

# Load CUDA extension inline
module = load_inline(
    name='my_cuda',
    cpp_sources='torch::Tensor my_op(torch::Tensor x);',
    cuda_sources='''
    __global__ void kernel(...) { ... }
    torch::Tensor my_op(torch::Tensor x) { ... }
    ''',
    functions=['my_op'],
)

# Setup.py integration
setup(
    name='my_extension',
    ext_modules=[
        CUDAExtension('my_cuda', ['src/my_cuda.cpp', 'src/my_cuda_kernel.cu']),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

---

## 60.4 Environment Variables Reference

| Variable | Purpose |
|----------|---------|
| CUDA_VISIBLE_DEVICES | Restrict visible GPUs |
| TORCH_HOME | Cache directory for models/data |
| TORCH_NUM_THREADS | Intra-op parallelism |
| TORCH_SHOW_DISPATCH_TRACE | Show dispatch decisions |
| TORCH_LOGS | Enable various logging |
| TORCHINDUCTOR_CACHE_DIR | Inductor cache location |
| TORCH_DISTRIBUTED_DEBUG | Debug distributed training |
| PYTORCH_CUDA_ALLOC_CONF | Memory allocator config |

---

## 60.5 Logging

```python
# torch._logging
import torch._logging
torch._logging.set_logs(dynamo=True, aot=True, inductor=True)

# Or via environment variable
# TORCH_LOGS="+dynamo,aot,inductor"
```

---

## 60.6 Benchmarking

```python
from torch.utils.benchmark import Timer, Compare

t = Timer(stmt='model(input)',
          setup='import torch; model = torch.nn.Linear(100, 100); input = torch.randn(32, 100)')
result = t.timeit(100)
print(result)
# <torch.utils.benchmark.utils.common.Measurement object>
# median: 50.2 us
```
