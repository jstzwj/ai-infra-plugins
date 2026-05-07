# 17 - Python Bindings (`nccl4py`) and NCCL EP Expert Parallelism Extension

## Primary source files

- `sources/nccl/bindings/nccl4py/README.md`
- `sources/nccl/bindings/nccl4py/setup.py`
- `sources/nccl/bindings/nccl4py/pyproject.toml`
- `sources/nccl/bindings/ir/*`
- `sources/nccl/contrib/nccl_ep/README.md`
- `sources/nccl/contrib/nccl_ep/nccl_ep.cc`
- `sources/nccl/contrib/nccl_ep/ep_test.cu`
- `sources/nccl/contrib/nccl_ep/ep_bench.cu`
- `sources/nccl/contrib/nccl_ep/ep_test.py`

## nccl4py overview

`nccl4py` provides Python bindings for NCCL with both low-level Cython bindings and a higher-level
Pythonic API for collective operations.

Requirements from the README:

| Component | Requirement |
|---|---|
| CUDA Toolkit | CUDA 12.x or 13.x |
| NCCL Library | matching CUDA package (`nvidia-nccl-cu12` or `nvidia-nccl-cu13`) |
| Python | 3.10+ |

Experimental Cython support:

```python
from nccl.bindings cimport cynccl
```

The `cynccl.pxd` file is included for direct Cython integration.

## nccl4py development setup

Set CUDA path:

```bash
export CUDA_HOME=/usr/local/cuda
```

Build with Makefile and `uv`:

```bash
cd sources/nccl/bindings/nccl4py
make dev
make build
make clean
```

Manual setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[cu12]   # CUDA 12.x
# or
pip install -e .[cu13]   # CUDA 13.x
pip install build
python -m build
```

The Makefile detects CUDA version from `CUDA_HOME`, installs CUDA-specific dependencies, and builds Cython
extensions.

## Bindings/IR

`bindings/ir` contains wrappers around NCCL device headers for IR/device integration. Files include:

- `nccl_device_wrapper.h`
- `nccl_device_wrapper__impl.h`
- CMake/Make build files.

Use this area when integrating NCCL device functionality into compiler/IR flows rather than normal Python
host bindings.

## NCCL EP overview

NCCL EP is a high-performance NCCL API extension for Mixture-of-Experts Expert Parallelism communication.
It provides dispatch and combine primitives implemented on top of NCCL Device API using:

- LSA (Load-Store Accessible) operations for local/NVLink communication,
- GIN (GPU-Initiated Networking) operations for RDMA/network communication.

It targets MoE token dispatch and expert output combine patterns in modern sparse LLMs.

## NCCL EP algorithms

| Algorithm | Target | Communication pattern |
|---|---|---|
| Low-Latency (LL) | small batch / inference | direct point-to-point all-to-all with experts |
| High-Throughput (HT) | training / prefill / large batch | hierarchical NVLink intra-node aggregation + RDMA inter-node |

HT mode leverages Hopper features such as warp-specialized pipelines and TMA operations according to the
README.

## NCCL EP key features

- Staged execution in LL mode through send-only flag.
- Automatic tuning of buffer sizes, queue pairs, and channels.
- Restricted type-conversion/scaling support.
- C and Python APIs.
- Benchmark and test tools: `ep_test`, `ep_bench`.

## NCCL EP dependencies

From README:

| Component | Version/notes |
|---|---|
| CUDA | 13+ |
| NCCL | 2.29+ with Device API and GIN support |
| MPI | any OpenMPI/MPICH-style MPI for multi-process launch |
| GPU | Hopper H100 or Blackwell tested |

## NCCL EP environment setup

```bash
export COMPUTE_CAP=<discovered_compute_cap>   # e.g. 90 for H100
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export NCCL_HOME=/path/to/nccl/build
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:${MPI_HOME}/bin:$PATH"
```

Build NCCL:

```bash
cd /path/to/nccl-source
make -j src.build BUILDDIR=${NCCL_HOME}
```

Build NCCL EP:

```bash
make -C contrib/nccl_ep MPI=1 BUILDDIR=${NCCL_HOME} \
  NVCC_GENCODE="-gencode=arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}"
```

Outputs include:

- `${NCCL_HOME}/lib/libnccl_ep.a`
- `${NCCL_HOME}/lib/libnccl_ep.so`
- `${NCCL_HOME}/include/nccl_ep.h`
- `${NCCL_HOME}/test/nccl_ep/ep_test`
- `${NCCL_HOME}/test/nccl_ep/ep_bench`

For multi-node RDMA GIN:

```bash
export NCCL_GIN_TYPE=3  # GDAKI
```

Debug:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## NCCL EP C API quick shape

```c
ncclEpCreateGroup(&ep_group, comm, &config, stream, alloc_fn, free_fn);
ncclEpGroupDestroy(ep_group, stream);

ncclEpCreateHandle(&handle, ep_group, topk_idx, local_tensors, num_local, config, stream);
ncclEpHandleDestroy(handle);

ncclEpDispatch(handle, inputs, num_in, outputs, num_out,
               local, num_local, send_only, config, stream);
ncclEpCombine(handle, inputs, num_in, outputs, num_out,
              local, num_local, send_only, config, stream);
ncclEpComplete(handle, config, stream);  // LL mode only
```

## NCCL EP Python API quick shape

Install:

```bash
pip install -e contrib/nccl_ep/python
```

Use:

```python
from nccl_ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY

nccl_lib = NCCLLibrary()
# nccl_lib.ncclEpDispatch(...)
# nccl_lib.ncclEpCombine(...)
```

## NCCL EP core data structures

### `ncclNDTensor_t`

Opaque tensor metadata/data-layout handle. Created with `ncclEpTensorCreate` and queried with getter
functions:

```c
ncclEpTensorGetData(tensor, &data);
ncclEpTensorGetSizes(tensor, &sizes, &ndim);
```

Pass `data = nullptr` for library-managed memory, or a non-null pointer for user-managed memory.

### `ncclEpGroup_t`

Created from an NCCL communicator. It manages the distributed EP configuration across ranks.

### `ncclEpHandle_t`

Represents prepared routing/metadata for dispatch/combine based on top-k indices and local tensors.

## NCCL EP tensor tags and dimensions

Notation from README:

- `B`: batch size
- `H`: hidden dimension
- `S`: scales dimension
- `L`: local experts
- `K`: top-k
- `R`: number of ranks
- `N(r)`: number of tokens targeting rank `r`

Tags:

| Tag | Meaning |
|---|---|
| `NCCL_EP_TENSOR_TAG_TOKENS` | token tensor |
| `NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS` | top-k weights |
| `NCCL_EP_TENSOR_TAG_TOPK_IDX` | top-k expert indices |
| `NCCL_EP_TENSOR_TAG_SCALES` | scaling tensor |
| `NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE` | device receive expert counter |
| `NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST` | host receive expert counter |

### LL mode, same datatype

| Operation | Tensor | Tag | Dims |
|---|---|---|---|
| Dispatch | Input | TOKENS | `[B x H]` |
| Dispatch | Output | TOKENS | `[L x R x B x H]` |
| Dispatch | Local | CNTR_D | `[L]` |
| Combine | Input | TOKENS | `[L x R x B x H]` |
| Combine | Output | TOKENS | `[B x H]` |
| Combine | Local | WEIGHTS | `[B x K]` |

### HT mode, same datatype, forward

| Operation | Tensor | Tag | Dims |
|---|---|---|---|
| Dispatch | Input | TOKENS | `[B x H]` |
| Dispatch | Input | WEIGHTS | `[B x K]` |
| Dispatch | Input | INDEX | `[B x K]` |
| Dispatch | Output | TOKENS | `[N(r) x H]` |
| Dispatch | Output | WEIGHTS | `[N(r) x K]` |
| Dispatch | Output | INDEX | `[N(r) x K]` |
| Combine | Input | TOKENS | `[N(r) x H]` |
| Combine | Output | TOKENS | `[B x H]` |

Backward pass adds weights to combine input/output as described in the README.

## NCCL EP test commands

Single-node:

```bash
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ht -t 4096 -d 7168
```

Multi-node example:

```bash
mpirun -np 16 \
  --map-by ppr:8:node \
  -x NCCL_GIN_TYPE=3 \
  -x LD_LIBRARY_PATH \
  ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168
```

## When to recommend NCCL EP

Recommend NCCL EP when the user is implementing MoE expert-parallel dispatch/combine and wants NCCL-native
Device API/GIN/LSA integration. Do not recommend it for ordinary data-parallel allreduce, where standard
NCCL collectives are simpler.
