# Apache TVM Reference - Chapter 15: MetaSchedule Auto-Tuning

This reference covers MetaSchedule, TVM's automated framework for searching optimal TensorIR schedules. MetaSchedule explores the space of possible schedule transformations for each operator in a model, evaluates their performance, and stores the best results for future use. It replaces the manual schedule authoring process with an automated, reproducible search.

---

## 15.1 Overview

### 15.1.1 Motivation

Writing optimal schedules by hand is time-consuming and error-prone. The number of possible schedules grows combinatorially with each transformation decision: tile sizes, loop order, memory hierarchy mapping, parallelization strategy, and more. MetaSchedule automates this process by:

1. **Generating** candidate schedules from a configurable set of rules
2. **Evaluating** each candidate by building and measuring its execution time
3. **Recording** the best candidates in a persistent database
4. **Replaying** the best schedule for production compilation

### 15.1.2 Architecture

MetaSchedule follows a modular architecture where each component can be customized independently:

```
+-------------------+
|   TaskScheduler   |  Orchestrates tuning across multiple tasks
+---------+---------+
          |
    +-----+------+
    | SearchTask |  A single tuning task (one PrimFunc)
    +-----+------+
          |
   +------+-------+
   |              |
+--+--+    +------+--+
|Space |    | Search  |
|Gen   |    |Strategy |
+--+--+    +------+--+
   |              |
   +------+-------+
          |
   +------+-------+    +-----------+
   | ScheduleRule |    |  Mutator  |
   +--+--+--+--+--+    +-----+-----+
      |  |  |  |             |
      v  v  v  v             v
   Candidate Schedules
          |
   +------+-------+
   |   Postproc   |
   +------+-------+
          |
   +------+-------+
   |   Builder    |  Compiles TIR to executable
   +------+-------+
          |
   +------+-------+
   |   Runner     |  Measures execution time
   +------+-------+
          |
   +------+-------+
   |   Database   |  Stores best results
   +------+-------+
          |
   +------+-------+
   |  CostModel   |  Predicts performance
   +------+-------+
```

---

## 15.2 Key Components

### 15.2.1 SearchStrategy

The search strategy determines how MetaSchedule explores the schedule space. It controls the balance between exploration (trying new strategies) and exploitation (refining known good strategies).

**Available strategies:**

| Strategy | Description |
|----------|-------------|
| `EvolutionarySearch` | Uses evolutionary algorithms (mutation + crossover) to explore the space |
| | Starts with rule-generated candidates and iteratively mutates them |

```python
from tvm.meta_schedule import search_strategy

# Create an evolutionary search strategy
strategy = search_strategy.EvolutionarySearch(
    num_trials_per_iter=64,    # Candidates per iteration
    num_trials_total=2048,     # Total search budget
    population_size=2048,      # Population for evolutionary search
    init_max_unfinished=256,   # Maximum unfinished initial candidates
    genetic_algo_iters=10,     # Number of evolutionary iterations
    max_fail_count=10,         # Max consecutive failures before stopping
    prob_mutate_idx=0.85,      # Probability of mutation vs crossover
)
```

**Configuration parameters for EvolutionarySearch:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_trials_per_iter` | 64 | Number of trials per search iteration |
| `num_trials_total` | 2048 | Total number of trials to run |
| `population_size` | 2048 | Size of the evolutionary population |
| `init_max_unfinished` | 256 | Max pending initial candidates |
| `genetic_algo_iters` | 10 | Evolutionary iterations |
| `max_fail_count` | 10 | Max consecutive failures |
| `prob_mutate_idx` | 0.85 | Mutation probability |

---

### 15.2.2 SpaceGenerator

The space generator creates the initial set of candidate schedules by applying schedule rules to the input PrimFunc.

**Available generators:**

| Generator | Description |
|-----------|-------------|
| `PostOrderApply` | Applies schedule rules in post-order (leaves first) traversal of the IR |

```python
from tvm.meta_schedule import space_generator

# Create a post-order space generator
space_gen = space_generator.PostOrderApply(
    sch_rules=[
        # List of ScheduleRule objects (see Section 15.3)
    ],
    postprocs=[
        # List of Postproc objects (see Section 15.4)
    ],
    mutator_probs={
        # Mutator -> probability mapping (see Section 15.5)
    },
)
```

**How PostOrderApply works:**

1. Traverse the PrimFunc's block structure in post-order
2. At each block, try all registered schedule rules
3. Each rule may produce zero or more candidate schedules (by calling sampling primitives)
4. The union of all candidate schedules forms the initial search space

---

### 15.2.3 ScheduleRule

Schedule rules define the transformation strategies that MetaSchedule can apply. Each rule examines a block and decides what transformations to apply.

**Rules are context-sensitive:** A rule can examine the block's structure, its surrounding loops, and its access patterns before deciding whether to apply. If a rule does not match the current block, it returns `None` and the generator moves to the next rule.

See Section 15.3 for detailed descriptions of each rule.

---

### 15.2.4 Mutator

Mutators modify existing candidate schedules to create new variants. They are used during the evolutionary search phase to explore neighborhoods of good schedules.

**Available mutators:**

| Mutator | Description |
|---------|-------------|
| `MutateTileSize` | Randomly changes tile sizes in the schedule |
| `MutateUnroll` | Randomly changes unroll factors |
| `MutateComputeLocation` | Moves a compute block to a different position |

```python
from tvm.meta_schedule.mutator import MutateTileSize, MutateUnroll, MutateComputeLocation

mutator_probs = {
    MutateTileSize(): 0.5,
    MutateUnroll(): 0.3,
    MutateComputeLocation(): 0.2,
}
```

---

### 15.2.5 Postproc

Post-processors run after all schedule rules have been applied. They perform final clean-up, verification, and rewriting.

See Section 15.4 for detailed descriptions of each post-processor.

---

### 15.2.6 Runner

The runner executes compiled modules and measures their performance.

**Available runners:**

| Runner | Description |
|--------|-------------|
| `LocalRunner` | Executes on the local machine |
| `RPCRunner` | Executes on a remote machine via TVM RPC |

```python
from tvm.meta_schedule import runner

# Local runner
local_runner = runner.LocalRunner(
    timeout_sec=30,          # Timeout per measurement
    max_workers=4,           # Parallel measurement workers
    evaluator_config=runner.EvaluatorConfig(
        number=3,            # Number of runs per measurement
        repeat=1,            # Number of repeat groups
        min_repeat_ms=100,   # Minimum time per repeat group
        enable_cpu_cache_flush=True,  # Flush CPU cache between runs
    ),
)

# RPC runner (for cross-compilation)
rpc_runner = runner.RPCRunner(
    rpc_config=runner.RPCConfig(
        tracker_host="127.0.0.1",
        tracker_port=9190,
        tracker_key="a100",
        session_timeout_sec=60,
    ),
    timeout_sec=30,
    max_workers=1,
    evaluator_config=runner.EvaluatorConfig(
        number=3,
        repeat=1,
        min_repeat_ms=100,
        enable_cpu_cache_flush=False,
    ),
)
```

**EvaluatorConfig parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `number` | 3 | Number of timing measurements per repeat |
| `repeat` | 1 | Number of measurement repeat groups |
| `min_repeat_ms` | 100 | Minimum duration (ms) for each repeat group |
| `enable_cpu_cache_flush` | `False` | Flush L1/L2 cache between measurements (CPU only) |
| `cool_down_interval_ms` | 0 | Cooldown between measurements (GPU only) |

---

### 15.2.7 Database

The database stores tuning results persistently, enabling reuse across compilation sessions.

**Available databases:**

| Database | Description |
|----------|-------------|
| `JSONDatabase` | Stores results in JSON files on disk |
| `MemoryDatabase` | Stores results in memory (lost on exit) |

```python
from tvm.meta_schedule.database import JSONDatabase, MemoryDatabase

# JSON database (persistent)
db = JSONDatabase(
    path_workload="workloads.json",
    path_tuning_record="tuning_records.json",
)

# Memory database (transient)
db = MemoryDatabase()

# Create a database that uses existing tuning data
db = JSONDatabase(
    path_workload="tuning_workloads.json",
    path_tuning_record="tuning_records.json",
)
```

**Database operations:**

```python
# Query the best schedule for a specific workload
from tvm.meta_schedule.database import TuningRecord

# Get the top-k records
records = db.get_top_k(workload, top_k=10)
for record in records:
    print(f"Record: {record.trace}")
    print(f"  Run time: {record.run_secs} seconds")

# Check if a workload has been tuned
has_record = db.has_workload(workload)
```

**TuningRecord fields:**

| Field | Type | Description |
|-------|------|-------------|
| `trace` | `tvm.ir.Trace` | The schedule trace (sequence of primitives) |
| `run_secs` | `List[float]` | Measured execution times |
| `workload` | `Workload` | The IRModule hash / identifier |
| `target` | `Target` | The compilation target |
| `args_info` | `List[ArgInfo]` | Information about function arguments |

---

### 15.2.8 CostModel

The cost model predicts the performance of a candidate schedule without actually building and running it. This speeds up the search by filtering out obviously bad candidates.

```python
from tvm.meta_schedule.cost_model import XGBoostModel

cost_model = XGBoostModel(
    num_warmup_samples=100,   # Samples before model kicks in
    num_features=4096,         # Number of features per candidate
)
```

**Available cost models:**

| Model | Description |
|-------|-------------|
| `XGBoostModel` | Gradient-boosted decision tree model using TIR features |
| `MLPCMModel` | Multi-layer perceptron cost model (experimental) |

---

### 15.2.9 FeatureExtractor

The feature extractor computes numerical features from TIR programs for use by the cost model.

```python
from tvm.meta_schedule.feature_extractor import PerStoreFeature

extractor = PerStoreFeature(
    max_workers=4,          # Parallel feature extraction
    store_buffer_size=2,    # Number of buffer features
    arith_intensity_pct=1,  # Arithmetic intensity percentiles
)
```

**Feature categories extracted by `PerStoreFeature`:**

| Category | Description | Dimensions |
|----------|-------------|------------|
| Buffer access features | Read/write patterns, strides, types | Per-buffer |
| Loop features | Trip counts, nesting depth, parallelism | Per-loop |
| Arithmetic features | Operation types and counts | Per-block |
| Memory hierarchy | Cache hit rates, bandwidth utilization | Global |
| Computation density | FLOPs per byte accessed | Global |

---

### 15.2.10 MeasureCallback

Callbacks invoked after each measurement round. Used for logging, progress tracking, and database updates.

```python
from tvm.meta_schedule.measure_callback import (
    MeasureCallback,
    SendToDatabase,
    UpdateCostModel,
)

callbacks = [
    SendToDatabase(),       # Store results in the database
    UpdateCostModel(),      # Update the cost model with new data
]
```

---

### 15.2.11 TaskScheduler

The task scheduler manages the allocation of tuning budget across multiple tasks (operators) in a model.

```python
from tvm.meta_schedule.task_scheduler import RoundRobin

scheduler = RoundRobin(
    tasks=[task1, task2, task3],
    trial_budgets=[500, 500, 500],   # Per-task budgets
)
```

**Available task schedulers:**

| Scheduler | Description |
|-----------|-------------|
| `RoundRobin` | Allocates equal budget to each task in round-robin order |
| `Gradient` | Allocates budget based on gradient of expected improvement |

---

### 15.2.12 Builder

The builder compiles TIR programs into executable modules for measurement.

```python
from tvm.meta_schedule.builder import LocalBuilder

builder = LocalBuilder(
    max_workers=4,           # Parallel build workers
    timeout_sec=30,          # Build timeout
)
```

---

## 15.3 Schedule Rules

### 15.3.1 AutoInline

Automatically inlines producer-consumer block pairs that meet certain criteria. This is typically the first rule applied and eliminates trivial intermediate computations.

```python
from tvm.meta_schedule.schedule_rule import AutoInline

rule = AutoInline(
    inline_into_producer=True,     # Allow inlining into producers
    inline_into_consumer=True,     # Allow inlining into consumers
    enable_inline_constraint=False, # Enable constraint checking
    inline_const_tensor=True,      # Inline constant tensor operations
    disallow_if_then_else=True,    # Do not inline conditionals
    require_injective=True,        # Require injective access patterns
    require_ordered=True,          # Require ordered iteration
)
```

**Inlining criteria:**
- The block must be element-wise (no reduction axes)
- The block has no side effects
- The access pattern is injective (each output element depends on exactly one input element)
- Inlining does not create nested conditionals

### 15.3.2 AddRFactor

Adds a reduction factor to a reduction block, splitting the reduction axis and creating a parallel dimension.

```python
from tvm.meta_schedule.schedule_rule import AddRFactor

rule = AddRFactor(
    max_jobs_per_core=16,  # Maximum parallel jobs per CPU core
    max_outer_product_factor=64,  # Maximum outer product factor
)
```

**Effect:** For a reduction `for k in range(N): C[i, j] += A[i, k] * B[k, j]`, AddRFactor splits `k` into `k_o` and `k_i`, where `k_i` is the new serial reduction and `k_o` is a parallel dimension. After the parallel reduction, a finalization block sums the partial results.

### 15.3.3 CrossThreadReduction

Performs reduction across GPU threads using warp-level or block-level reduction primitives.

```python
from tvm.meta_schedule.schedule_rule import CrossThreadReduction

rule = CrossThreadReduction(
    thread_extents=[16, 32, 64, 128, 256, 512],  # Candidate thread counts
)
```

**Effect:** Binds the reduction axis to `threadIdx` and uses GPU shuffle or shared memory to perform the cross-thread reduction. This eliminates the need for atomic operations and enables parallel reduction.

### 15.3.4 MultiLevelTiling

The most important schedule rule for dense computation (matmul, conv2d). It applies multi-level loop tiling with optional caching at each level.

```python
from tvm.meta_schedule.schedule_rule import MultiLevelTiling

# GPU variant
gpu_tiling = MultiLevelTiling(
    structure="SSRSRS",    # S=spatial, R=reduce tile levels
    tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y", "threadIdx.x"],
    use_tensor_core=False,  # Enable Tensor Core (WMMA) tiling
    max_innermost_factor=64,
    vector_load_lens=[1, 2, 3, 4, 8, 16],
    reuse_read=schedule_rule.MultiLevelTiling.ReuseType(
        req="must",       # "must" or "no"
        levels=[1, 2],    # Cache at tile levels 1 and 2
        scope="shared",   # Storage scope for cached data
    ),
    reuse_write=schedule_rule.MultiLevelTiling.ReuseType(
        req="no",
        levels=[],
        scope="",
    ),
)

# CPU variant
cpu_tiling = MultiLevelTiling(
    structure="SSRSRS",
    tile_binds=None,       # No GPU thread binding
    max_innermost_factor=128,
    vector_load_lens=[1, 2, 4, 8, 16, 32],
    reuse_read=schedule_rule.MultiLevelTiling.ReuseType(
        req="must",
        levels=[1],
        scope="global",
    ),
    reuse_write=schedule_rule.MultiLevelTiling.ReuseType(
        req="no",
        levels=[],
        scope="",
    ),
)
```

**Structure string interpretation:**

| Character | Meaning | Description |
|-----------|---------|-------------|
| `S` | Spatial tile | Tile a spatial (non-reduction) dimension |
| `R` | Reduce tile | Tile a reduction dimension |

For `structure="SSRSRS"`:
- `S` (1st): Outer spatial tile -- mapped to `blockIdx.y`
- `S` (2nd): Outer spatial tile -- mapped to `blockIdx.x`
- `R` (1st): Outer reduction tile -- corresponds to pipeline stages
- `S` (3rd): Inner spatial tile -- mapped to `threadIdx.y`
- `R` (2nd): Inner reduction tile -- serial reduction within a thread
- `S` (4th): Innermost spatial tile -- vectorization / unrolling target

### 15.3.5 ParallelizeVectorizeUnroll

Applies parallelization, vectorization, and unrolling annotations to loops based on their position in the loop nest.

```python
from tvm.meta_schedule.schedule_rule import ParallelizeVectorizeUnroll

rule = ParallelizeVectorizeUnroll(
    vectorize_probs=[0.0, 0.0, 0.0, 1.0],   # Probability of vectorizing at each level
    unroll_probs=[0.0, 0.0, 1.0, 0.0],        # Probability of unrolling at each level
    unroll_max_steps=[0, 0, 64, 0],            # Max unroll step at each level
    max_vectorize_extend_len=64,               # Max vector width
)
```

### 15.3.6 RandomComputeLocation

Moves a compute block to a random position in the loop nest. This is used to explore different compute locations for blocks that have flexibility in where they are placed.

```python
from tvm.meta_schedule.schedule_rule import RandomComputeLocation

rule = RandomComputeLocation()
```

---

## 15.4 Post-Processors

Post-processors run after all schedule rules have been applied. They perform final transformations and verification.

### 15.4.1 RewriteCooperativeFetch

Rewrites shared memory loads to use cooperative fetching, where multiple threads in a thread block cooperate to load a tile of data.

```python
from tvm.meta_schedule.postproc import RewriteCooperativeFetch

postproc = RewriteCooperativeFetch()
```

**Effect:** When a cache_read block loads data into shared memory, this post-processor ensures that the loading is distributed across all threads in the thread block. Each thread loads a subset of the tile, maximizing memory bandwidth utilization.

### 15.4.2 RewriteParallelVectorizeUnroll

Rewrites parallel, vectorize, and unroll annotations based on the final loop structure. Ensures that only loops that benefit from these annotations receive them.

```python
from tvm.meta_schedule.postproc import RewriteParallelVectorizeUnroll

postproc = RewriteParallelVectorizeUnroll()
```

### 15.4.3 RewriteReductionBlock

Rewrites reduction blocks that have been decomposed to ensure correct initialization and accumulation patterns.

```python
from tvm.meta_schedule.postproc import RewriteReductionBlock

postproc = RewriteReductionBlock()
```

### 15.4.4 RewriteUnboundBlock

Handles blocks that are not bound to any loop (i.e., they have no surrounding iteration). This can happen when a block's output is a scalar or when the block represents a single operation.

```python
from tvm.meta_schedule.postproc import RewriteUnboundBlock

postproc = RewriteUnboundBlock()
```

### 15.4.5 VerifyGPUCode

Verifies that the scheduled program satisfies GPU constraints. Rejects candidates that violate hardware limits.

```python
from tvm.meta_schedule.postproc import VerifyGPUCode

postproc = VerifyGPUCode()
```

**Constraints checked:**
- Max threads per block (typically 1024)
- Max shared memory per block (typically 48KB or 96KB)
- Max threads per dimension
- Max vector width

---

## 15.5 Tuning Workflow

### 15.5.1 Step-by-Step Process

The complete MetaSchedule tuning workflow consists of five steps:

**Step 1: Extract tasks from the IRModule**

```python
import tvm
from tvm import relax
from tvm.meta_schedule import extract_tasks

# Import and optimize a model
mod = relax.get_pipeline("zero")(imported_mod)

# Extract tunable tasks
tasks = extract_tasks(mod, target=target, params=params)
print(f"Extracted {len(tasks)} tuning tasks")
for i, task in enumerate(tasks):
    print(f"  Task {i}: {task.task_name}")
```

**Step 2: Generate search space**

```python
from tvm.meta_schedule import space_generator, search_strategy
from tvm.meta_schedule.schedule_rule import (
    AutoInline,
    MultiLevelTiling,
    ParallelizeVectorizeUnroll,
    CrossThreadReduction,
)
from tvm.meta_schedule.postproc import (
    RewriteCooperativeFetch,
    RewriteParallelVectorizeUnroll,
    VerifyGPUCode,
)

# Configure the space generator
space_gen = space_generator.PostOrderApply(
    sch_rules=[
        AutoInline(),
        MultiLevelTiling(
            structure="SSRSRS",
            tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y", "threadIdx.x"],
            reuse_read=...,
        ),
        CrossThreadReduction(),
        ParallelizeVectorizeUnroll(),
    ],
    postprocs=[
        RewriteCooperativeFetch(),
        RewriteParallelVectorizeUnroll(),
        VerifyGPUCode(),
    ],
    mutator_probs={
        MutateTileSize(): 0.5,
        MutateUnroll(): 0.3,
    },
)
```

**Step 3: Sample and evaluate candidates**

```python
from tvm.meta_schedule.tune import tune_tasks

database = tune_tasks(
    tasks=tasks,
    task_scheduler=RoundRobin(),
    space_generator=space_gen,
    search_strategy=EvolutionarySearch(
        num_trials_total=2000,
    ),
    builder=LocalBuilder(max_workers=4),
    runner=LocalRunner(max_workers=4),
    database=JSONDatabase(
        path_workload="workloads.json",
        path_tuning_record="records.json",
    ),
    cost_model=XGBoostModel(),
    measure_callbacks=[SendToDatabase()],
)
```

**Step 4: Store best results**

Results are automatically stored in the database during tuning. The database persists across sessions:

```python
# Results are stored as TuningRecords containing:
# - The schedule trace (sequence of primitives)
# - Measured execution times
# - Target information
# - Workload hash
```

**Step 5: Apply best schedule**

```python
from tvm.meta_schedule import apply_database

# Apply the best schedules from the database
mod = apply_database(mod, database, target=target)

# Build the optimized module
exec = relax.build(mod, target=target)
```

### 15.5.2 Quick Tuning with Pipeline

For convenience, TVM provides a one-shot tuning pipeline:

```python
from tvm import relax

# Tune and build in one step
mod = relax.get_pipeline("static_shape_tuning")(
    mod,
    target=target,
    params=params,
    tuning_logs_directory="./tuning_logs",
)
```

**Pipeline configuration:**

```python
# Customize the tuning pipeline
from tvm.meta_schedule.tune import TuneConfig

config = TuneConfig(
    strategy="evolutionary",    # Search strategy
    num_trials_per_iter=64,
    max_trials_per_task=2048,
    max_trials_global=32768,
    builder=LocalBuilder(max_workers=4),
    runner=LocalRunner(timeout_sec=30),
)
```

---

## 15.6 Configuration Options and Tuning Parameters

### 15.6.1 Global Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_trials_global` | 32768 | Total tuning budget across all tasks |
| `max_trials_per_task` | 2048 | Per-task tuning budget |
| `num_trials_per_iter` | 64 | Candidates evaluated per iteration |
| `builder_timeout_sec` | 30 | Build timeout per candidate |
| `runner_timeout_sec` | 30 | Execution timeout per candidate |
| `max_fail_count` | 10 | Max consecutive failures before stopping a task |

### 15.6.2 Target-Specific Configurations

**NVIDIA GPU (A100):**

```python
from tvm.meta_schedule.schedule_rule import MultiLevelTiling

gpu_rules = [
    AutoInline(),
    MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y", "threadIdx.x"],
        use_tensor_core=True,          # Enable Tensor Core
        max_innermost_factor=64,
        vector_load_lens=[1, 2, 4, 8],
        reuse_read=MultiLevelTiling.ReuseType(
            req="must",
            levels=[1, 2],
            scope="shared",
        ),
        reuse_write=MultiLevelTiling.ReuseType(
            req="must",
            levels=[2],
            scope="local",
        ),
    ),
    CrossThreadReduction(
        thread_extents=[16, 32, 64, 128, 256],
    ),
    ParallelizeVectorizeUnroll(
        vectorize_probs=[0.0, 0.0, 0.0, 1.0],
        unroll_probs=[0.0, 0.0, 1.0, 0.0],
        unroll_max_steps=[0, 0, 64, 0],
    ),
]
```

**CPU (x86 with AVX-512):**

```python
cpu_rules = [
    AutoInline(),
    MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=128,
        vector_load_lens=[1, 2, 4, 8, 16, 32, 64],
        reuse_read=MultiLevelTiling.ReuseType(
            req="must",
            levels=[1],
            scope="global",
        ),
    ),
    AddRFactor(max_jobs_per_core=16),
    ParallelizeVectorizeUnroll(
        vectorize_probs=[0.0, 0.0, 1.0],
        unroll_probs=[0.0, 1.0, 0.0],
        unroll_max_steps=[0, 64, 0],
    ),
]
```

### 15.6.3 Database Reuse

To reuse tuning results across compilation sessions:

```python
from tvm.meta_schedule.database import JSONDatabase

# First compilation session: tune and save
db = JSONDatabase(
    path_workload="./tuning/workloads.json",
    path_tuning_record="./tuning/records.json",
)

# ... run tuning ...

# Second compilation session: reuse results
db_reuse = JSONDatabase(
    path_workload="./tuning/workloads.json",
    path_tuning_record="./tuning/records.json",
)

# Check how many tasks have existing tuning data
for task in tasks:
    if db_reuse.has_workload(task.workload):
        print(f"Task '{task.task_name}' has cached tuning data")
    else:
        print(f"Task '{task.task_name}' needs tuning")
```

### 15.6.4 Logging and Monitoring

```python
import logging

# Enable detailed MetaSchedule logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tvm.meta_schedule")
logger.setLevel(logging.DEBUG)

# Monitor tuning progress
class ProgressCallback:
    def __init__(self):
        self.best_time = float("inf")
        self.trial_count = 0

    def __call__(self, records):
        for record in records:
            self.trial_count += 1
            avg_time = sum(record.run_secs) / len(record.run_secs)
            if avg_time < self.best_time:
                self.best_time = avg_time
                print(f"Trial {self.trial_count}: New best = {avg_time * 1e6:.2f} us")
```

---

## 15.7 Integration with Relax Pipeline

### 15.7.1 Standard Tuning Flow

The typical end-to-end flow for tuning a model:

```python
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Step 1: Import model
import torch
import torchvision
model = torchvision.models.resnet18(pretrained=True).eval()
sample_input = torch.randn(1, 3, 224, 224)

mod = from_exported_program(
    torch.export.export(model, (sample_input,))
)

# Step 2: Define target
target = tvm.target.Target("nvidia/nvidia-a100")

# Step 3: Apply default optimization pipeline
mod = relax.get_pipeline("zero")(mod)

# Step 4: Extract tasks
from tvm.meta_schedule import extract_tasks
tasks = extract_tasks(mod, target=target)

# Step 5: Tune (with existing database if available)
from tvm.meta_schedule.tune import tune_tasks
database = tune_tasks(
    tasks=tasks,
    target=target,
    max_trials_global=10000,
    database=JSONDatabase("w.json", "r.json"),
)

# Step 6: Compile with tuned schedules
mod = relax.get_pipeline("static_shape_tuning")(mod, database=database)
exec = relax.build(mod, target=target)

# Step 7: Deploy
vm = relax.VirtualMachine(exec, tvm.cuda(0))
result = vm["main"](tvm.nd.array(sample_input.numpy(), device=tvm.cuda(0)))
```

### 15.7.2 Using Pre-tuned Database

```python
from tvm.meta_schedule.database import JSONDatabase

# Load a pre-tuned database (e.g., from a previous tuning session)
db = JSONDatabase(
    path_workload="pre_tuned_workloads.json",
    path_tuning_record="pre_tuned_records.json",
)

# Compile directly using cached tuning results
mod = relax.get_pipeline("static_shape_tuning")(
    mod,
    target=target,
    database=db,
)
exec = relax.build(mod, target=target)
```

---

## 15.8 Advanced Topics

### 15.8.1 Custom Schedule Rules

You can implement custom schedule rules by subclassing `ScheduleRule`:

```python
from tvm.meta_schedule.schedule_rule import ScheduleRule
import tvm

class MyCustomRule(ScheduleRule):
    """A custom schedule rule that applies a specific transformation."""

    def initialize_with_tune_context(self, context):
        """Called once before tuning begins."""
        self.target = context.target

    def apply(self, sch, block):
        """Apply the rule to a block. Return None if the rule does not match."""
        # Check if this block matches the expected pattern
        block_stmt = sch.get(block).stmt

        # Example: only apply to blocks with 3 iteration axes
        if len(block_stmt.iter_vars) != 3:
            return None

        # Apply transformations
        loops = sch.get_loops(block)
        i, j, k = loops

        # Tile with specific factors
        i_o, i_i = sch.split(i, factors=[None, 32])
        j_o, j_i = sch.split(j, factors=[None, 32])

        # Reorder
        sch.reorder(i_o, j_o, i_i, j_i, k)

        return [sch]
```

### 15.8.2 Custom Post-Processors

```python
from tvm.meta_schedule.postproc import Postproc

class MyCustomPostproc(Postproc):
    """Custom post-processor that applies final transformations."""

    def initialize_with_tune_context(self, context):
        self.target = context.target

    def apply(self, sch):
        """Apply post-processing. Return False to reject the schedule."""
        # Example: verify that all loops have reasonable extents
        for block_rv in sch.get_all_blocks():
            for loop_rv in sch.get_loops(block_rv):
                sref = sch.get(loop_rv)
                extent = sref.stmt.extent
                if isinstance(extent, int.IntImm) and extent > 4096:
                    return False  # Reject this schedule
        return True  # Accept
```

### 15.8.3 Multi-Target Tuning

```python
# Tune for multiple targets simultaneously
targets = [
    tvm.target.Target("nvidia/nvidia-a100"),
    tvm.target.Target("nvidia/nvidia-t4"),
    tvm.target.Target("cpu"),
]

all_databases = {}
for target in targets:
    print(f"Tuning for {target}")
    tasks = extract_tasks(mod, target=target)
    db = tune_tasks(
        tasks=tasks,
        target=target,
        max_trials_global=5000,
        database=JSONDatabase(
            f"workloads_{str(target)}.json",
            f"records_{str(target)}.json",
        ),
    )
    all_databases[str(target)] = db
```

---

## 15.9 Summary

| Component | Purpose | Key Implementations |
|-----------|---------|-------------------|
| `SearchStrategy` | Explore schedule space | `EvolutionarySearch` |
| `SpaceGenerator` | Generate initial candidates | `PostOrderApply` |
| `ScheduleRule` | Transform blocks | `AutoInline`, `MultiLevelTiling`, `CrossThreadReduction`, `ParallelizeVectorizeUnroll`, `AddRFactor`, `RandomComputeLocation` |
| `Mutator` | Mutate existing candidates | `MutateTileSize`, `MutateUnroll`, `MutateComputeLocation` |
| `Postproc` | Post-process schedules | `RewriteCooperativeFetch`, `RewriteParallelVectorizeUnroll`, `VerifyGPUCode`, `RewriteReductionBlock`, `RewriteUnboundBlock` |
| `Runner` | Measure performance | `LocalRunner`, `RPCRunner` |
| `Database` | Store results | `JSONDatabase`, `MemoryDatabase` |
| `CostModel` | Predict performance | `XGBoostModel` |
| `FeatureExtractor` | Extract TIR features | `PerStoreFeature` |
| `Builder` | Compile candidates | `LocalBuilder` |
| `TaskScheduler` | Allocate tuning budget | `RoundRobin`, `Gradient` |
| `MeasureCallback` | Post-measurement hooks | `SendToDatabase`, `UpdateCostModel` |
