# Chapter 10: Runtime Cache Module (`triton.runtime.cache`)

## Cache Architecture

Triton caches compiled kernels to avoid recompilation. The cache stores:
- Compiled PTX/AMDGPU assembly
- CUBIN/HSACO binaries
- Compilation metadata
- Autotune results

### Cache Location

Default: `~/.triton/cache/`
Override: `TRITON_HOME` environment variable

```python
import os
os.environ['TRITON_HOME'] = '/custom/path'
# Cache will be at /custom/path/.triton/cache/
```

## CacheManager Classes

### `CacheManager` (ABC)
Abstract base class for cache backends.

```python
class CacheManager(ABC):
    @abstractmethod
    def get_file(self, filename) -> Optional[str]

    @abstractmethod
    def put(self, data, filename, binary=True) -> str

    @abstractmethod
    def get_group(self, filenames) -> Optional[List[str]]

    @abstractmethod
    def put_group(self, data_list, filenames, binary=True) -> List[str]
```

### `FileCacheManager`
Default file-based cache implementation.

```python
# Files are stored atomically using temporary files
# Key is computed from kernel source, types, and backend
```

Features:
- Atomic writes (temp file + rename)
- JSON group support for multiple related files
- Binary and text storage

### `RemoteCacheBackend` (ABC)
Abstract base for remote cache backends.

### `RedisRemoteCacheBackend`
Redis-based remote cache.

```python
# Configured via environment variables:
# TRITON_CACHE_REMOTE_URL - Redis URL
# TRITON_CACHE_REMOTE_USERNAME
# TRITON_CACHE_REMOTE_PASSWORD
```

### `RemoteCacheManager`
Combines remote backend with local file cache for materialization.

## Cache Key Functions

### `get_cache_key(src, backend, backend_options, env_vars) -> str`
Generate comprehensive cache key from all compilation inputs.

```python
# Key includes:
# - Source code hash
# - Backend hash (GPU target)
# - Compilation options hash
# - Environment variable hash
```

### `make_so_cache_key(version_hash, signature, constants, ids, **kwargs) -> str`
Generate cache key for shared objects (launcher code).

### `triton_key() -> str`
Generate Triton-specific key including version and component hashes.

## Cache Control

### Environment Variables

| Variable | Effect |
|----------|--------|
| `TRITON_ALWAYS_COMPILE=1` | Force recompilation, ignore cache |
| `TRITON_CACHE_DIR` | Override cache directory |
| `TRITON_HOME` | Override base Triton directory |
| `TRITON_CACHE_REMOTE_URL` | Redis URL for remote cache |
| `TRITON_CACHE_REMOTE_USERNAME` | Redis username |
| `TRITON_CACHE_REMOTE_PASSWORD` | Redis password |

### Cache Inspection Hook

```python
def inspect_stages(self, stages, options, language, capability):
    # Inspect or modify compilation stages
    for stage_name, (ir_fn, save_ir) in stages.items():
        print(f"Stage: {stage_name}")
    return stages

triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```

## Kernel Override System

Override compiled kernels for debugging:

```bash
# Step 1: Enable dumping
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=/tmp/triton_dump

# Step 2: Run kernel (dumps IR and PTX)
python your_script.py

# Step 3: Copy and modify
cp -r /tmp/triton_dump/<hash> /tmp/triton_override/<hash>
# Edit IR files in /tmp/triton_override/<hash>/

# Step 4: Override and re-run
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=/tmp/triton_override
python your_script.py
```

## Cache Invalidation

Cache keys include:
- Source code content
- Argument types (specialization)
- Backend target (GPU architecture)
- Compilation options
- Environment variables (selected via `_env_vars` parameter)
- Triton version

Any change to these triggers recompilation.
