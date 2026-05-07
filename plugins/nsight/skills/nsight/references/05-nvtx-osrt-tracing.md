# NVTX, OS Runtime Libraries, OpenMP, and Syscall Tracing Reference

This document provides comprehensive reference material for tracing features in NVIDIA Nsight Systems that cover OpenMP, OS Runtime Libraries (OSRT), Linux syscalls, and the NVIDIA Tools Extension (NVTX) API. These tracing capabilities allow developers to instrument, observe, and analyze application behavior at the runtime library, kernel boundary, and user-annotation levels.

---

## Table of Contents

1. [OpenMP Trace](#openmp-trace)
   - [Configuration and Requirements](#openmp-configuration)
   - [Functions Traced (OMPT Callbacks)](#openmp-functions-traced)
2. [OS Runtime Libraries Trace](#os-runtime-libraries-trace)
   - [Overview](#osrt-overview)
   - [Enabling OS Runtime Libraries Tracing](#enabling-osrt)
   - [Locking a Resource](#locking-a-resource)
   - [Limitations](#osrt-limitations)
3. [OS Runtime Libraries Trace Filters](#osrt-trace-filters)
4. [OS Runtime Default Function List](#osrt-default-function-list)
   - [Libc System Call Wrappers](#libc-syscall-wrappers)
   - [POSIX Threads](#posix-threads)
   - [I/O Functions](#io-functions)
   - [Miscellaneous Functions](#miscellaneous-functions)
5. [Syscall Trace (Experimental)](#syscall-trace)
   - [Requirements](#syscall-requirements)
   - [Enabling Syscall Trace](#enabling-syscall)
   - [Behavior Notes](#syscall-behavior)
6. [NVTX Trace](#nvtx-trace)
   - [NVTX Overview](#nvtx-overview)
   - [Using NVTX in Your Application](#using-nvtx)
   - [NVTX Best Practices](#nvtx-best-practices)
   - [NVTX Payloads and Counters (Preview)](#nvtx-payloads-counters)
   - [NVTX Domains and Categories](#nvtx-domains-categories)
   - [NVTX Range Naming and Domain Filtering](#nvtx-domain-filtering)
   - [Using NVTX for Capture Ranges](#nvtx-capture-ranges)

---

## OpenMP Trace

<a id="openmp-configuration"></a>

### Configuration and Requirements

Nsight Systems for Linux is capable of capturing information about OpenMP events. This functionality is built on the **OpenMP Tools Interface (OMPT)**. Full support is available only for runtime libraries that implement the tools interface defined in **OpenMP 5.0** or greater.

#### Compiler and Runtime Notes

- **LLVM OpenMP runtime library** partially implements the tools interface.
- **PGI compiler <= 20.4**: Add the `-mp=libomp` switch to use the LLVM OpenMP runtime and enable OMPT-based tracing.
- **Clang**: Ensure the LLVM OpenMP runtime library you link to was compiled with the tools interface enabled.

<a id="openmp-functions-traced"></a>

### Functions Traced (OMPT Callbacks)

Only a subset of the OMPT callbacks are processed. The complete list of supported callbacks:

| Callback | Description |
|---|---|
| `ompt_callback_parallel_begin` | Parallel region entry |
| `ompt_callback_parallel_end` | Parallel region exit |
| `ompt_callback_sync_region` | Synchronization region events |
| `ompt_callback_task_create` | Task creation events |
| `ompt_callback_task_schedule` | Task scheduling events |
| `ompt_callback_implicit_task` | Implicit task events |
| `ompt_callback_master` | Master region events |
| `ompt_callback_reduction` | Reduction operation events |
| `ompt_callback_cancel` | Cancellation events |
| `ompt_callback_mutex_acquire` | Mutex acquisition start |
| `ompt_callback_mutex_acquired` | Mutex acquisition completion |
| `ompt_callback_mutex_released` | Mutex release |
| `ompt_callback_work` | Worksharing constructs |
| `ompt_callback_dispatch` | Dispatch events |
| `ompt_callback_flush` | Flush events |

> **Note:** The raw OMPT events are used to generate ranges indicating the runtime of OpenMP operations and constructs. The tool does not expose the raw OMPT callbacks directly; instead, it synthesizes meaningful time ranges from them.

---

## OS Runtime Libraries Trace

<a id="osrt-overview"></a>

### Overview

On Linux, OS runtime libraries can be traced to gather information about low-level userspace APIs. This traces the system call wrappers and thread synchronization interfaces exposed by the C runtime and POSIX Threads (pthread) libraries. This does not perform a complete runtime library API trace, but instead focuses on the functions that can take a long time to execute, or could potentially cause your thread to be unscheduled from the CPU while waiting for an event to complete.

**OS runtime trace is not available for Windows targets.**

OS runtime tracing complements and enhances sampling information by:

1. **Visualizing hardware interactions** -- When the process is communicating with the hardware, controlling resources, performing multi-threading synchronization, or interacting with the kernel scheduler.

2. **Adding additional thread states** -- By correlating how OS runtime library traces affect the thread scheduling:
   - **Waiting** -- The thread is not scheduled on a CPU. It is inside an OS runtime libraries trace and is believed to be waiting on the firmware to complete a request.
   - **In OS runtime library function** -- The thread is scheduled on a CPU and inside an OS runtime libraries trace. If the trace represents a system call, the process is likely running in kernel mode.

3. **Collecting backtraces for long OS runtime library calls** -- This provides a way to gather blocked-state backtraces, allowing you to gain more context about why the thread was blocked for so long, while avoiding unnecessary overhead for short events.

4. **Collecting file access data** -- For API calls that interact with files. This helps in identifying performance bottlenecks related to file I/O operations and provides insights into how file access patterns affect overall application performance.
   - File access flags and mode information is collected.
   - File access bytes copied information is collected.

> **Note:** File access data collection is not enabled by default.

<a id="enabling-osrt"></a>

### Enabling OS Runtime Libraries Tracing

From Nsight Systems:

- **CLI**: Use the `-t`, `--trace` option with the `osrt` parameter.

  ```bash
  nsys profile -t osrt my_application
  ```

- **GUI**: Select the **Collect OS runtime libraries trace** checkbox.

You can also use **Skip if shorter than**. This will skip calls shorter than the given threshold. Enabling this option will improve performance as well as reduce noise on the timeline. **We strongly encourage you to skip OS runtime library calls shorter than 1 microsecond.**

### Locking a Resource

The functions listed below receive special treatment. If the tool detects that the resource is already acquired by another thread and will induce a blocking call, Nsight Systems always traces it. Otherwise, it is never traced.

| Function | Description |
|---|---|
| `pthread_mutex_lock` | POSIX mutex lock |
| `pthread_rwlock_rdlock` | POSIX read-write lock (read) |
| `pthread_rwlock_wrlock` | POSIX read-write lock (write) |
| `pthread_spin_lock` | POSIX spinlock |
| `sem_wait` | POSIX semaphore wait |

Note that even if a call is determined as potentially blocking, there is a chance that it may not actually block after a few cycles have elapsed. The call will still be traced in this scenario.

<a id="osrt-limitations"></a>

### Limitations

1. **Syscall wrappers only** -- Nsight Systems only traces syscall wrappers exposed by the C runtime. It is not able to trace syscalls invoked through assembly code.

2. **Sampling dependency** -- Additional thread states, as well as backtrace collection on long calls, are only enabled if sampling is turned on.

3. **Backtrace configuration** -- It is not possible to configure the depth and duration threshold when collecting backtraces. Currently, only OS runtime library calls longer than 80 microseconds will generate a backtrace with a maximum of 24 frames. This limitation will be removed in a future version of the product.

4. **Compiler flag requirement** -- It is required to compile your application and libraries with the `-funwind-tables` compiler flag in order for Nsight Systems to unwind the backtraces correctly.

---

## OS Runtime Libraries Trace Filters

<a id="osrt-trace-filters"></a>

The OS runtime libraries tracing is limited to a select list of functions. It also depends on the version of the C runtime linked to the application. Only functions from the default function list (see below) will be traced.

---

## OS Runtime Default Function List

<a id="osrt-default-function-list"></a>

The following sections provide the complete lists of functions that are traced by the OS Runtime Libraries trace feature by default.

<a id="libc-syscall-wrappers"></a>

### Libc System Call Wrappers

The following libc system call wrapper functions are traced:

```
accept  accept4  acct  alarm  arch_prctl  bind  bpf  brk  chroot
clock_nanosleep  connect  copy_file_range  creat  creat64  dup  dup2
dup3  epoll_ctl  epoll_pwait  epoll_wait  fallocate  fallocate64
fcntl  fdatasync  flock  fork  fsync  ftruncate  futex  ioctl
ioperm  iopl  kill  killpg  listen  membarrier  mlock  mlock2
mlockall  mmap  mmap64  mount  move_pages  mprotect  mq_notify
mq_open  mq_receive  mq_send  mq_timedreceive  mq_timedsend  mremap
msgctl  msgget  msgrcv  msgsnd  msync  munmap  nanosleep
nfsservctl  open  open64  openat  openat64  pause  pipe  pipe2
pivot_root  poll  ppoll  prctl  pread  pread64  preadv  preadv2
preadv64  process_vm_readv  process_vm_writev  pselect6  ptrace
pwrite  pwrite64  pwritev  pwritev2  pwritev64  read  readv
reboot  recv  recvfrom  recvmmsg  recvmsg  rt_sigaction
rt_sigqueueinfo  rt_sigsuspend  rt_sigtimedwait  sched_yield
seccomp  select  semctl  semget  semop  semtimedop  send
sendfile  sendfile64  sendmmsg  sendmsg  sendto  shmat  shmctl
shmdt  shmget  shutdown  sigaction  sigsuspend  sigtimedwait
socket  socketpair  splice  swapoff  swapon  sync
sync_file_range  syncfs  tee  tgkill  tgsigqueueinfo  tkill
truncate  umount2  unshare  uselib  vfork  vhangup  vmsplice
wait  wait3  wait4  waitid  waitpid  write  writev  _sysctl
```

<a id="posix-threads"></a>

### POSIX Threads

The following POSIX Threads (pthread) functions are traced:

```
pthread_barrier_wait    pthread_cancel             pthread_cond_broadcast
pthread_cond_signal     pthread_cond_timedwait     pthread_cond_wait
pthread_create          pthread_join               pthread_kill
pthread_mutex_lock      pthread_mutex_timedlock    pthread_mutex_trylock
pthread_rwlock_rdlock   pthread_rwlock_timedrdlock pthread_rwlock_timedwrlock
pthread_rwlock_tryrdlock pthread_rwlock_trywrlock  pthread_rwlock_wrlock
pthread_spin_lock       pthread_spin_trylock       pthread_timedjoin_np
pthread_tryjoin_np      pthread_yield              sem_timedwait
sem_trywait             sem_wait
```

<a id="io-functions"></a>

### I/O Functions

The following I/O library functions are traced:

```
aio_fsync       aio_fsync64     aio_suspend     aio_suspend64
fclose          fcloseall       fflush          fflush_unlocked
fgetc           fgetc_unlocked  fgets           fgets_unlocked
fgetwc          fgetwc_unlocked fgetws          fgetws_unlocked
flockfile       fopen           fopen64         fputc
fputc_unlocked  fputs           fputs_unlocked  fputwc
fputwc_unlocked fputws          fputws_unlocked fread
fread_unlocked  freopen         freopen64       ftrylockfile
fwrite          fwrite_unlocked getc            getc_unlocked
getdelim        getline         getw            getwc
getwc_unlocked  lockf           lockf64         mkfifo
mkfifoat        posix_fallocate posix_fallocate64 putc
putc_unlocked   putwc           putwc_unlocked
```

<a id="miscellaneous-functions"></a>

### Miscellaneous Functions

```
forkpty  popen  posix_spawn  posix_spawnp  sigwait  sigwaitinfo
sleep    system usleep
```

---

## Syscall Trace (Experimental)

<a id="syscall-requirements"></a>

### Requirements

Nsight Systems for Linux and Nsight Systems Embedded Platforms Edition are capable of tracing Linux system calls in kernel space. This feature uses Linux's **eBPF** technology.

- **Linux kernel**: Version 5.11 or newer
- **Kernel configuration**: Built with `CONFIG_DEBUG_INFO_BTF` enabled (this is the default on most major Linux distributions)
- **Capabilities**: The `nsys` process requires `CAP_BPF` and `CAP_PERFMON` capabilities (alternatively, `CAP_SYS_ADMIN` or root privileges). See the capabilities man page for details.

To check if the target system meets the requirements:

```bash
# Check kernel version
uname -r

# Check if BTF is enabled
ls /sys/kernel/btf/vmlinux
```

<a id="enabling-syscall"></a>

### Enabling Syscall Trace

**CLI** -- Add the `--syscall` option to the `nsys start` or `nsys profile` commands (setting `syscall` in the `--trace` option is deprecated and will be ignored).

The following values are supported:

| Value | Description |
|---|---|
| `none` | No syscall tracing (default) |
| `process-tree` | Collect syscalls for the profiled application process and its child processes |
| `pid-namespace` | Collect syscalls made by all processes in the current PID namespace and its child namespaces. This is very close to how other features work in system-wide mode (e.g., inside a container, tracing will be limited to this container) |

**GUI** -- Select the **Collect syscall trace** checkbox. This is currently equivalent to the `--syscall=process-tree` option.

<a id="syscall-behavior"></a>

### Behavior Notes

- Only syscalls running **1000 nanoseconds or more** are traced.
- Long-running syscalls (more than **80 microseconds**) are also collected with their backtraces.

### Example Commands

```bash
# Trace syscalls for the profiled process and its children
nsys profile --syscall=process-tree my_application

# Trace syscalls for all processes in the current PID namespace
nsys profile --syscall=pid-namespace my_application

# Combine with other trace sources
nsys profile --trace=cuda,osrt --syscall=process-tree my_application
```

### Syscall Trace vs. OS Runtime Trace

| Feature | OS Runtime Trace | Syscall Trace |
|---|---|---|
| **Mechanism** | Function interception | eBPF kernel tracing |
| **Scope** | libc/libpthread functions only | All system calls from any code |
| **Static linking** | Not supported | Supported |
| **Overhead** | Low (per-call interception) | Very low (kernel-level) |
| **Kernel requirements** | Standard | Linux 5.11+, eBPF, capabilities |
| **Backtraces** | Supported (calls > 80 us) | Supported (calls > 80 us) |
| **Minimum duration** | Configurable (skip threshold) | 1000 ns (fixed) |
| **Maturity** | Stable | Experimental |

---

## NVTX Trace

<a id="nvtx-overview"></a>

### NVTX Overview

The **NVIDIA Tools Extension Library (NVTX)** is a powerful mechanism that allows users to manually instrument their application. Nsight Systems can then collect the information and present it on the timeline.

Nsight Systems supports **version 3.0** of the NVTX specification.

The following features are supported:

#### Domains

| Function | Description |
|---|---|
| `nvtxDomainCreate()` | Create a new NVTX domain |
| `nvtxDomainDestroy()` | Destroy an NVTX domain |
| `nvtxDomainRegisterString()` | Register a string in a domain |

#### Push-Pop Ranges (Nested Ranges)

Nested ranges that start and end in the same thread.

| Function | Description |
|---|---|
| `nvtxRangePush()` | Push a range onto the stack |
| `nvtxRangePushEx()` | Push an extended range onto the stack |
| `nvtxRangePop()` | Pop the most recent range from the stack |
| `nvtxDomainRangePushEx()` | Push an extended range in a specific domain |
| `nvtxDomainRangePop()` | Pop a range in a specific domain |

#### Start-End Ranges (Global Ranges)

Ranges that are global to the process and are not restricted to a single thread.

| Function | Description |
|---|---|
| `nvtxRangeStart()` | Start a range |
| `nvtxRangeStartEx()` | Start an extended range |
| `nvtxRangeEnd()` | End a range |
| `nvtxDomainRangeStartEx()` | Start an extended range in a specific domain |
| `nvtxDomainRangeEnd()` | End a range in a specific domain |

#### Marks

| Function | Description |
|---|---|
| `nvtxMark()` | Insert an instantaneous mark |
| `nvtxMarkEx()` | Insert an extended mark |
| `nvtxDomainMarkEx()` | Insert an extended mark in a specific domain |

#### Thread Names

| Function | Description |
|---|---|
| `nvtxNameOsThread()` | Assign a name to an OS thread |

#### Categories

| Function | Description |
|---|---|
| `nvtxNameCategory()` | Assign a name to a category |
| `nvtxDomainNameCategory()` | Assign a name to a category in a specific domain |

<a id="using-nvtx"></a>

### Using NVTX in Your Application

To use NVTX in your application, follow these steps:

1. **Include the header** -- Add the following include in your source code:

   ```c
   #include "nvtx3/nvToolsExt.h"
   ```

   The `nvtx3` directory is located in the Nsight Systems package in the `Target-<architecture>/nvtx/include` directory and is available via GitHub at [NVIDIA/NVTX](https://github.com/NVIDIA/NVTX).

2. **Link the library** -- Add the following compiler flag:

   ```
   -ldl
   ```

3. **Add NVTX API calls** -- For example, try adding `nvtxRangePush("main")` at the beginning of the `main()` function, and `nvtxRangePop()` just before the return statement at the end.

   ```c
   int main(int argc, char** argv) {
       nvtxRangePush("main");

       // Application code here

       nvtxRangePop();
       return 0;
   }
   ```

4. **RAII wrapper (C++)** -- For convenience in C++ code, consider adding a wrapper that implements the RAII (Resource Acquisition Is Initialization) pattern, which would guarantee that every range gets closed.

   ```cpp
   class NvtxRange {
   public:
       NvtxRange(const char* name) { nvtxRangePushA(name); }
       ~NvtxRange() { nvtxRangePop(); }
   };

   void my_function() {
       NvtxRange range("my_function");
       // Range automatically pops when `range` goes out of scope
   }
   ```

5. **GUI settings** -- In the project settings, select the **Collect NVTX trace** checkbox.

6. **Hotkey markers** -- By enabling the **Insert NVTX Marker hotkey** option, it is possible to add NVTX markers to a running non-console application by pressing the **F11** key. These will appear in the report under the NVTX Domain named **"HotKey markers."**

### Enabling NVTX Tracing

**CLI**:

```bash
nsys profile --trace=nvtx my_application
```

**GUI**: Select the **Collect NVTX trace** checkbox in the project settings.

<a id="nvtx-best-practices"></a>

### NVTX Best Practices

- **Low overhead when not profiling**: Typically, calls to NVTX functions can be left in the source code even if the application is not being built for profiling purposes, since the overhead is very low when the profiler is not attached.

- **Avoid annotating very small code pieces**: NVTX is not intended to annotate very small pieces of code that are being called very frequently. A good rule of thumb: if code being annotated usually takes less than **1 microsecond** to execute, adding an NVTX range around this code should be done carefully.

- **Match range annotations carefully**: If many ranges are opened but not closed, Nsight Systems has no meaningful way to visualize it. A rule of thumb is to not have more than a couple dozen ranges open at any point in time. Nsight Systems does not support reports with many unclosed ranges.

---

<a id="nvtx-payloads-counters"></a>

### NVTX Payloads and Counters (Preview)

NVTX Extended Payloads and NVTX Counters increase the flexibility of NVTX annotations by allowing users to pass arbitrary structured data to NVTX events. This then allows users to define the layout of this data in the Nsight Systems UI without additional data conversion.

Key concepts:

- **Payloads** -- Allow attaching arbitrary key-value data to NVTX ranges and marks. The payload schema is defined by the user.
- **Counters** -- Allow recording numeric values over time, which can be displayed as graphs on the timeline.

For more information, see the NVTX documentation.

---

<a id="nvtx-domains-categories"></a>

### NVTX Domains and Categories

NVTX domains enable scoping of annotations. Unless specified differently, all events and annotations are in the **default domain**. Additionally, categories can be used to group events.

Nsight Systems gives the user the ability to include or exclude NVTX events from a particular domain. This can be especially useful if you are profiling across multiple libraries and are only interested in NVTX events from some of them.

Categories that are set by the user will be recognized and displayed in the GUI.

<a id="nvtx-domain-filtering"></a>

### NVTX Range Naming and Domain Filtering

#### Domain Filtering via CLI

| Option | Description |
|---|---|
| `--nvtx-domain-include=<domain>` | Include only NVTX events from the specified domain(s) |
| `--nvtx-domain-exclude=<domain>` | Exclude NVTX events from the specified domain(s) |

Multiple domains can be specified as a comma-separated list.

```bash
# Only capture annotations from the specified domains
nsys profile --nvtx-domain-include="DataLoader,Compute" --trace=nvtx ./my_application

# Capture all domains except the specified ones
nsys profile --nvtx-domain-exclude="Debug,Logging" --trace=nvtx ./my_application
```

When neither option is specified, all NVTX domains are captured.

#### Domain Filtering via GUI

In the GUI, use the NVTX domain selection screen to include or exclude specific domains from the trace view.

1. **Color coding** -- NVTX ranges can carry color information that is respected in the timeline visualization.
2. **Category grouping** -- Categories assigned via `nvtxNameCategory()` or `nvtxDomainNameCategory()` are displayed in the NVTX rows, allowing visual grouping of related events.

---

<a id="nvtx-capture-ranges"></a>

### Using NVTX for Capture Ranges

Nsight Systems supports using NVTX ranges to define the start and stop of data collection. This is particularly useful for long-running applications where you only want to profile a specific phase of execution.

**CLI option**: Use `--capture-range=nvtx` to start and stop profiling based on NVTX range boundaries.

When combined with the NVTX instrumentation in your application, this allows you to:

- **Reduce report size** -- Only capture data during the annotated region of interest.
- **Avoid startup/shutdown noise** -- Skip initialization and finalization phases.
- **Focus on specific algorithm phases** -- Profile only the computation-heavy parts of your application.

Example:

```bash
# Only capture data between NVTX range start and end
nsys profile --capture-range=nvtx --capture-range-end=stop my_application
```

This requires your application to be instrumented with NVTX range annotations that define the capture region boundaries.

---

## Quick Reference: CLI Options Summary

| Option | Values | Description |
|---|---|---|
| `-t osrt` / `--trace=osrt` | `osrt` | Enable OS Runtime Libraries tracing |
| `--trace=omp` | `omp` | Enable OpenMP tracing |
| `--syscall` | `none`, `process-tree`, `pid-namespace` | Enable syscall tracing (experimental) |
| `--trace=nvtx` | `nvtx` | Enable NVTX tracing |
| `--nvtx-domain-include` | comma-separated domain names | Include specific NVTX domains |
| `--nvtx-domain-exclude` | comma-separated domain names | Exclude specific NVTX domains |
| `--capture-range` | `nvtx`, etc. | Use NVTX to define capture boundaries |

---

## Additional Notes

### OSRT and Sampling Interaction

OS runtime libraries tracing works best when combined with CPU sampling. The additional thread states (Waiting, In OS runtime library function) are only available when sampling is enabled. Similarly, backtrace collection for long OS runtime library calls requires sampling to be active.

### Compiler Flags for Best Results

For optimal tracing and backtrace resolution, compile your application with the following flags:

```bash
-g                  # Include debug information
-funwind-tables     # Generate unwind tables for backtrace resolution
-fno-omit-frame-pointer  # (Optional) Enable frame pointer-based unwinding
```

### Overhead Considerations

| Feature | Overhead | Notes |
|---|---|---|
| OSRT tracing | Moderate | Use the "Skip if shorter than" threshold to reduce noise |
| Syscall tracing (eBPF) | Low | Only syscalls >= 1000ns are traced |
| NVTX tracing | Very low (when not profiling) | Minimal overhead during profiling |
| OpenMP tracing | Low | Only processes a subset of OMPT callbacks |

### Platform Availability

| Feature | Linux x86_64 | Linux Arm (SBSA) | Windows | QNX |
|---|---|---|---|---|
| OpenMP Trace | Yes | Yes | No | No |
| OS Runtime Libraries Trace | Yes | Yes | No | Yes (limited) |
| Syscall Trace (eBPF) | Yes (kernel >= 5.11) | Yes (kernel >= 5.11) | No | No |
| NVTX Trace | Yes | Yes | Yes | Yes |
