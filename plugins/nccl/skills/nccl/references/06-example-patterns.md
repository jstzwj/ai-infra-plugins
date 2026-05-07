# 06 - NCCL Example Patterns

## Primary source directory

- `sources/nccl/docs/examples/README.md`
- `sources/nccl/docs/examples/01_communicators/`
- `sources/nccl/docs/examples/02_point_to_point/`
- `sources/nccl/docs/examples/03_collectives/`
- `sources/nccl/docs/examples/04_user_buffer_registration/`
- `sources/nccl/docs/examples/05_symmetric_memory/`
- `sources/nccl/docs/examples/06_device_api/`
- `sources/nccl/docs/examples/common/`

The examples are educational templates. They are not designed as peak-performance implementations; use
`nccl-tests` for performance benchmarking.

## Build examples

From NCCL source root:

```bash
make -j examples
make -j examples MPI=1
```

Against an existing build:

```bash
cd docs/examples
make NCCL_HOME=/path/to/nccl MPI=1
```

Run threaded examples:

```bash
NTHREADS=8 ./example_binary
```

Run MPI examples:

```bash
mpirun -np 8 ./example_binary
```

## Example family selection

| Folder | Pattern | API focus | When to use as template |
|---|---|---|---|
| `01_communicators/01_multiple_devices_single_process` | one process manages all GPUs | `ncclCommInitAll` | simplest single-node test |
| `01_communicators/02_one_device_per_pthread` | one pthread per GPU | `ncclCommInitRank` | single-process threaded runtime |
| `01_communicators/03_one_device_per_process_mpi` | one MPI process per GPU | `ncclCommInitRank`, MPI broadcast | multi-node/production-like training |
| `02_point_to_point/01_ring_pattern` | ring send/recv | `ncclSend`, `ncclRecv`, groups | pairwise dataflow and deadlock-free P2P |
| `03_collectives/01_allreduce` | all ranks reduce and distribute | `ncclAllReduce` | gradient averaging/global sum |
| `04_user_buffer_registration/01_allreduce` | register buffers once | `ncclMemAlloc`, `ncclCommRegister` | repeated collectives on same buffers |
| `05_symmetric_memory/01_allreduce` | symmetric windows | `ncclCommWindowRegister` | low-latency/high-bandwidth symmetric collectives |
| `06_device_api/01_allreduce_lsa` | custom kernel allreduce | `ncclDevCommCreate`, LSA | fused local collectives |
| `06_device_api/02_alltoall_gin` | network-only device all-to-all | GIN | multi-node GPU-initiated communication baseline |
| `06_device_api/03_alltoall_hybrid` | LSA local + GIN remote | teams, GIN, LSA | optimized multi-node custom communication |

## Pattern: single process manages multiple GPUs

Use `ncclCommInitAll`.

Skeleton:

```cpp
int num_gpus = 0;
cudaGetDeviceCount(&num_gpus);

std::vector<ncclComm_t> comms(num_gpus);
ncclCommInitAll(comms.data(), num_gpus, NULL);

for (int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(i);
  cudaStreamCreate(&streams[i]);
  cudaMalloc(&buffers[i], bytes);
}

ncclGroupStart();
for (int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(i);
  ncclAllReduce(send[i], recv[i], count, ncclFloat32, ncclSum, comms[i], streams[i]);
}
ncclGroupEnd();

for (int i = 0; i < num_gpus; ++i) cudaStreamSynchronize(streams[i]);
```

Advantages:

- Lowest setup complexity.
- No MPI/pthread bootstrap.
- Useful for minimal examples and local tests.

Limitations:

- Single node only.
- One process owns all GPU contexts.
- Not representative of most distributed training launchers.

## Pattern: one pthread per GPU

Use `ncclGetUniqueId` once, shared across threads, then each thread calls:

```cpp
cudaSetDevice(local_device);
ncclCommInitRank(&comm, nranks, id, rank);
```

Use barriers to ensure:

1. Unique ID is generated before threads initialize.
2. Threads do not free shared state too early.
3. All threads reach cleanup.

This pattern is useful for single-node runtimes that use thread-level parallelism instead of one process
per GPU.

## Pattern: one MPI process per GPU

Typical production shape:

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &world);

ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

int localRank = compute_local_rank_somehow();
cudaSetDevice(localRank);

ncclComm_t comm;
ncclCommInitRank(&comm, world, id, rank);
```

Key points:

- NCCL does not provide its own launcher.
- MPI is only used for process launch and bootstrap in examples.
- Rank-to-GPU mapping is application responsibility.
- Multi-node support comes from the launcher and NCCL network transports.

## Pattern: P2P ring

Use send to next rank and receive from previous rank:

```cpp
int prev = (rank - 1 + nranks) % nranks;
int next = (rank + 1) % nranks;

ncclGroupStart();
ncclSend(sendbuf, count, dtype, next, comm, stream);
ncclRecv(recvbuf, count, dtype, prev, comm, stream);
ncclGroupEnd();
```

Why grouping is required: each P2P op may wait for matching progress. Without grouping, a rank can block
on a send while no matching recv is posted.

## Pattern: AllReduce correctness example

Expected sum for ranks contributing rank value:

```text
sum_{r=0}^{nranks-1} r = nranks * (nranks - 1) / 2
```

Steps:

1. Allocate and initialize send buffers with rank-specific values.
2. Group `ncclAllReduce(... ncclSum ...)` over local comms.
3. Synchronize streams.
4. Copy back and verify every element equals expected sum.

## Pattern: user buffer registration

```cpp
ncclMemAlloc((void**)&d_send, bytes);
ncclMemAlloc((void**)&d_recv, bytes);

ncclCommRegister(comm, d_send, bytes, &send_handle);
ncclCommRegister(comm, d_recv, bytes, &recv_handle);

ncclAllReduce(d_send, d_recv, count, ncclFloat32, ncclSum, comm, stream);

ncclCommDeregister(comm, send_handle);
ncclCommDeregister(comm, recv_handle);
ncclMemFree(d_send);
ncclMemFree(d_recv);
```

Use when repeated operations use the same buffers and registration overhead/resource usage matters.

## Pattern: symmetric memory

```cpp
ncclMemAlloc(&buffer, bytes);
ncclCommWindowRegister(comm, buffer, bytes, &win, NCCL_WIN_COLL_SYMMETRIC);

ncclAllReduce(buffer, buffer, count, ncclFloat32, ncclSum, comm, stream);

ncclCommWindowDeregister(comm, win);
ncclMemFree(buffer);
```

Use when all ranks have consistent memory layouts and the target system supports optimized symmetric
collective protocols.

## Pattern: Device API custom kernel

Host setup:

```cpp
ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
ncclCommQueryProperties(comm, &props);

ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = cta_count;

ncclDevComm_t devComm;
ncclDevCommCreate(comm, &reqs, &devComm);

kernel<<<cta_count, threads, 0, stream>>>(windows..., devComm);
```

Device-side code uses `nccl_device` helpers for teams, barriers, windows, and communication.

## Common utility patterns

The `docs/examples/common` directory contains shared helpers for advanced examples:

- CUDA/NCCL error-checking macros.
- MPI utility functions.
- Thread/bootstrap utilities.
- Common allocation and validation helpers.

A typical NCCL macro prints file/line, error string, and failed operation before exiting. In production
libraries, return errors to the caller instead of calling `exit`, but keep the same amount of diagnostic
context.

## Example troubleshooting

| Symptom | Likely issue | First check |
|---|---|---|
| communicator init hangs | ID not broadcast to all ranks, rank count mismatch, device not set | rank logs before/after `ncclCommInitRank` |
| AllReduce returns wrong data | stream not synchronized before validation | `cudaStreamSynchronize` after NCCL call |
| P2P ring hangs | sends/recvs not grouped or peer mismatch | `ncclGroupStart/End`, prev/next formulas |
| MPI example uses wrong GPU | local rank mapping wrong | print hostname, rank, local rank, CUDA device |
| registered-buffer example fails | allocation/registration mismatch or early free | use `ncclMemAlloc`, check deregister order |
| Device API example exits early | property check failed | print `deviceApiSupport`, `ginType`, `nLsaTeams` |
