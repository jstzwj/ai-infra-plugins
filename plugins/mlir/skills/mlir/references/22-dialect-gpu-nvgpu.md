# MLIR GPU & NVGPU Dialects

## GPU Dialect

### GPU Module

```mlir
gpu.module @kernel_module {
  gpu.func @kernel(%arg0: memref<10xf32>) kernel {
    %tid = gpu.thread_id  x
    %ntid = gpu.block_dim x
    // kernel body
    gpu.return
  }
}
```

### GPU Launch

```mlir
// Launch kernel
gpu.launch blocks(%bx, %by, %bz) in (%grid_x, %grid_y, %grid_z) =
              threads(%tx, %ty, %tz) in (%block_x, %block_y, %block_z)
  args(%arg0: memref<10xf32>) {
  // kernel body
  gpu.terminator
}

// Launch function
gpu.launch_func @kernel_module::@kernel
  blocks in (%grid_x, %grid_y, %grid_z)
  threads in (%block_x, %block_y, %block_z)
  args(%arg0: memref<10xf32>)
```

### Thread Indexing

```mlir
%tid_x = gpu.thread_id  x : index
%tid_y = gpu.thread_id  y : index
%tid_z = gpu.thread_id  z : index

%bid_x = gpu.block_id  x : index
%bid_y = gpu.block_id  y : index
%bid_z = gpu.block_id  z : index

%bdim_x = gpu.block_dim x : index
%bdim_y = gpu.block_dim y : index
%bdim_z = gpu.block_dim z : index

%gdim_x = gpu.grid_dim  x : index
%gdim_y = gpu.grid_dim  y : index
%gdim_z = gpu.grid_dim  z : index
```

### Synchronization

```mlir
// Block-level barrier
gpu.barrier

// Wait for async operations
gpu.wait async [%token] : !gpu.async.token

// Async memcpy
%token = gpu.memcpy async [%dep] %dst, %src : memref<10xf32>, memref<10xf32>
```

### Memory Operations

```mlir
// Dynamic shared memory
%smem = gpu.dynamic_shared_memory : !gpu.ptr<i8>

// Host registration
gpu.host_register %ptr : memref<10xf32>
gpu.host_unregister %ptr : memref<10xf32>

// Alloc on GPU
%buf = gpu.alloc () : memref<10xf32>
%buf_dyn = gpu.alloc (%n) : memref<?xf32>

// Dealloc
gpu.dealloc %buf : memref<10xf32>

// Set default device
gpu.set_default_device %device : i32

// Shuffle (intra-warp)
%shfl, %pred = gpu.shuffle %val, %offset, %width xor : f32
```

### Complete GPU Operations Reference

| Operation | Description |
|-----------|-------------|
| `gpu.module` | GPU kernel container |
| `gpu.func` | GPU kernel function |
| `gpu.return` | Return from GPU func |
| `gpu.launch` | Launch inline kernel |
| `gpu.launch_func` | Launch named kernel |
| `gpu.terminator` | Terminate launch body |
| `gpu.thread_id` | Thread index |
| `gpu.block_id` | Block index |
| `gpu.block_dim` | Block dimension |
| `gpu.grid_dim` | Grid dimension |
| `gpu.barrier` | Block barrier |
| `gpu.wait` | Wait async operations |
| `gpu.memcpy` | Async memory copy |
| `gpu.memset` | Async memory set |
| `gpu.alloc` | GPU memory allocation |
| `gpu.dealloc` | GPU memory deallocation |
| `gpu.host_register` | Register host memory |
| `gpu.host_unregister` | Unregister host memory |
| `gpu.set_default_device` | Set GPU device |
| `gpu.shuffle` | Warp shuffle |
| `gpu.printf` | Device printf |
| `gpu.dynamic_shared_memory` | Dynamic shared memory |
| `gpu.group_id` | Workgroup ID |
| `gpu.num_groups` | Number of workgroups |
| `gpu.subgroup_id` | Subgroup ID |
| `gpu.num_subgroups` | Number of subgroups |
| `gpu.subgroup_size` | Subgroup size |
| `gpu.subgroup_reduce` | Subgroup reduction |

## NVGPU Dialect

NVIDIA GPU-specific operations:

```mlir
// MMA (Matrix Multiply-Accumulate)
%D = nvgpu.mma.sync (%A, %B, %C)
    : (tensor<16x8xf16>, tensor<16x8xf16>, tensor<16x8xf32>) -> tensor<16x8xf32>

// Warp matrix load/store
%matrix = nvgpu.warp_load %mem[%i, %j] : memref<16x8xf16> -> tensor<16x8xf16>
nvgpu.warp_store %matrix, %mem[%i, %j] : tensor<16x8xf16>, memref<16x8xf16>

// TMA (Tensor Memory Accelerator) load
%desc = nvgpu.tma.create.descriptor %mem {boxDim = [64]} : !nvgpu.tma.descriptor
nvgpu.tma.load %desc, %coord : !nvgpu.tma.descriptor

// LdMatrix (load matrix fragment)
%frag = nvgpu.ldmatrix %smem[%offset] {num = 4 : i32, transpose} : memref<16x16xf16> -> vector<4x2xf16>
```

## AMDGPU Dialect

AMD GPU-specific operations:

```mlir
// MFMA (Matrix Fused Multiply-Add)
%result = amdgpu.mfma %a, %b, %c
    {blkM = 16, blkN = 16, blkK = 4, m = 4, n = 4, k = 4}
    : vector<4xf16>, vector<4xf16>, vector<4xf32> -> vector<4xf32>

// WMMA (Wavefront Matrix Multiply-Accumulate)
%frag = amdgpu.wmma %a, %b, %c : vector<4xf16>, vector<4xf16>, vector<4xf32> -> vector<4xf32>
```

## GPU Compilation Pipeline

```c++
void buildGPUPipeline(OpPassManager &pm) {
  // Lower to GPU dialect
  pm.addPass(createConvertLinalgToGPUPass());

  // Kernel outlining
  pm.addPass(createGpuKernelOutliningPass());

  // Lower GPU to NVVM/SPIR-V
  pm.nest<gpu::GPUModuleOp>().addPass(createConvertGPUToNVVMPass());
  // or
  pm.nest<gpu::GPUModuleOp>().addPass(createConvertGPUToSPIRVPass());

  // Lower to LLVM
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertArithToLLVMPass());
}
```
