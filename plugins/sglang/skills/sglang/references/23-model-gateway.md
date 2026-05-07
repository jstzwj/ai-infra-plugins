# SGLang Model Gateway Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Deployment Modes](#deployment-modes)
6. [Load Balancing Strategies](#load-balancing-strategies)
7. [Data Parallelism Routing](#data-parallelism-routing)
8. [Health Checking](#health-checking)
9. [Reliability and Flow Control](#reliability-and-flow-control)
10. [API Compatibility](#api-compatibility)
11. [Tokenizer Management](#tokenizer-management)
12. [Reasoning Parser Integration](#reasoning-parser-integration)
13. [Tool Call Parsing](#tool-call-parsing)
14. [MCP Integration](#mcp-integration)
15. [Service Discovery (Kubernetes)](#service-discovery-kubernetes)
16. [History and Data Connectors](#history-and-data-connectors)
17. [WASM Middleware](#wasm-middleware)
18. [Language Bindings](#language-bindings)
19. [Security and Authentication](#security-and-authentication)
20. [Production Deployment](#production-deployment)
21. [Configuration Reference](#configuration-reference)
22. [Monitoring and Observability](#monitoring-and-observability)
23. [Troubleshooting](#troubleshooting)

---

## Overview

SGLang Model Gateway (SMG) is a high-performance, Rust-based model-routing gateway purpose-built for large-scale LLM deployments. It serves as the unified control and data plane for managing fleets of inference workers, balancing traffic across heterogeneous protocols (HTTP, gRPC, OpenAI-compatible), and providing enterprise-grade reliability, observability, and security.

The gateway is written entirely in Rust for maximum throughput and minimal latency overhead. It is deeply optimized for the SGLang serving runtime (SRT) but can route to any OpenAI-compatible backend, including OpenAI, xAI, Gemini, vLLM, and TensorRT-LLM.

### Key Capabilities

- **Unified control plane** for registering, monitoring, and orchestrating regular, prefill, and decode workers across heterogeneous model fleets
- **Multi-protocol data plane** that routes traffic across HTTP, PD (prefill/decode), gRPC, and OpenAI-compatible backends with shared reliability primitives
- **Industry-first gRPC pipeline** with native Rust tokenization, reasoning parsers, and tool-call execution for high-throughput, OpenAI-compatible serving; supports both single-stage and PD topologies
- **Inference Gateway Mode** (`--enable-igw`) dynamically instantiates multiple router stacks (HTTP regular/PD, gRPC) and applies per-model policies for multi-tenant deployments
- **Conversation and response connectors** centralize chat history inside the router so the same context can be reused across models and MCP loops without leaking data to upstream vendors
- **Enterprise privacy**: agentic multi-turn `/v1/responses`, native MCP client (STDIO/HTTP/SSE/Streamable), and history storage all operate within the router boundary
- **Reliability core**: retries with jitter, worker-scoped circuit breakers, token-bucket rate limiting with queuing, background health checks, and cache-aware load monitoring
- **Comprehensive observability**: 40+ Prometheus metrics, OpenTelemetry distributed tracing, structured logging, and request ID propagation

### Performance Highlights

The cache-aware routing policy in SMG significantly improves performance for workloads with shared prefixes:

| Metric | Without Cache-Aware | With Cache-Aware SMG |
|--------|---------------------|----------------------|
| Throughput (token/s) | 82,665 | 158,596 (+92%) |
| Cache Hit Rate | 20% | 75% (+275%) |

These benchmarks were measured with multiple long prefix groups across 8x A100 80GB GPUs at dp-size=8, as reported in the SGLang v0.4 blog.

### Binary Aliases

The gateway can be invoked under three names, all pointing to the same binary:

```
sgl-model-gateway   # Full name
smg                 # Short alias
amg                 # Alternative short alias
```

### Version Information

```bash
# Simple version
smg --version
# Output: sgl-model-gateway 0.3.2

# Verbose version with build details
smg --version-verbose
# Output includes: build time, platform, git commit, compiler versions
```

---

## Architecture

### High-Level Architecture

The gateway is organized into a control plane and a data plane, with shared observability and security layers.

```
                           SGLang Model Gateway Architecture
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         Client Requests                             │
  │              (OpenAI-compatible / SGLang / gRPC)                    │
  └────────────────────────────┬────────────────────────────────────────┘
                               │
  ┌────────────────────────────▼────────────────────────────────────────┐
  │                        Security Layer                               │
  │     API Key Auth │ mTLS │ JWT/OIDC │ Control Plane RBAC             │
  └────────────────────────────┬────────────────────────────────────────┘
                               │
  ┌────────────────────────────▼────────────────────────────────────────┐
  │                      Middleware Pipeline                             │
  │   WASM │ Concurrency Limit │ Rate Limit │ Request ID │ CORS         │
  └────────────────────────────┬────────────────────────────────────────┘
                               │
  ┌────────────────────────────▼────────────────────────────────────────┐
  │                       Router Manager                                │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
  │  │ HTTP Router   │  │  gRPC Router │  │ OpenAI Router│              │
  │  │(Regular / PD) │  │(Single / PD) │  │   (Proxy)    │              │
  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
  └─────────┼─────────────────┼─────────────────┼───────────────────────┘
            │                 │                 │
  ┌─────────▼─────────────────▼─────────────────▼───────────────────────┐
  │                    Policy Registry                                  │
  │  cache_aware │ round_robin │ power_of_two │ random │ bucket │ ...   │
  └────────────────────────────┬────────────────────────────────────────┘
                               │
  ┌────────────────────────────▼────────────────────────────────────────┐
  │                    Worker Registry                                  │
  │   Health Checks │ Circuit Breakers │ Load Monitor │ Hash Ring      │
  └─────────┬────────────────┬───────────────────┬─────────────────────┘
            │                │                   │
  ┌─────────▼───────┐ ┌─────▼────────┐  ┌───────▼──────────┐
  │ HTTP Workers    │ │ gRPC Workers │  │ OpenAI Backends  │
  │ (SGLang/vLLM/  │ │ (SRT gRPC)   │  │ (OpenAI/xAI/    │
  │  TRT-LLM)      │ │              │  │  Gemini/...)     │
  └────────────────┘ └──────────────┘  └──────────────────┘
```

### Control Plane

The control plane manages worker lifecycle, job processing, and service discovery.

#### Worker Manager

The `WorkerManager` is responsible for discovering worker capabilities, tracking load, and registering/removing workers in the shared registry. When a worker is added, the manager:

1. Detects the connection mode (HTTP or gRPC) from the URL prefix
2. Queries worker metadata via `/server_info` and `/get_model_info` endpoints
3. Discovers data parallelism configuration for co-launched workers
4. Registers the worker in the `WorkerRegistry` with model ID, priority, cost, and type
5. Updates the policy registry to include the new worker in routing decisions

#### Job Queue

The `JobQueue` serializes asynchronous add/remove/update operations to avoid blocking the data plane. Key characteristics:

- **Background processing**: Worker registration, tokenizer loading, and MCP server initialization all run as background jobs
- **Status tracking**: Clients can poll `/workers/{worker_id}` to track job progress through `pending`, `processing`, and `completed` states
- **Workflow engine**: Typed workflow engines (`WorkflowEngines`) manage multi-step operations with subscriber notifications for logging and metrics

The queue configuration includes:

```rust
pub struct JobQueueConfig {
    pub max_retries: u32,      // Default: 3
    pub retry_delay_ms: u64,   // Default: 1000
}
```

#### Load Monitor

The `LoadMonitor` periodically samples worker loads and feeds data to load-aware policies (cache-aware, power-of-two). It runs on a configurable interval (`--worker-startup-check-interval-secs`, default 30 seconds) and tracks:

- Running request counts per worker
- Token throughput metrics
- Worker health status transitions

#### Health Checker

The health checker runs as a background task spawned by the `WorkerRegistry`. It probes workers on a configurable interval and updates worker health status, circuit breaker state, and router metrics. See the Health Checking section for full details.

#### Tokenizer Registry

The `TokenizerRegistry` manages dynamically registered tokenizers with async loading from HuggingFace or local paths. Tokenizers are loaded via the workflow engine (`TokenizerRegistration` step) and cached with two-level caching (L0 exact match, L1 prefix match).

### Data Plane

The data plane handles all request routing through three independent router implementations.

#### HTTP Router

The HTTP router handles two sub-modes:

- **Regular router**: Routes requests to single-stage workers with per-model policy overrides. Supports `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/classify`, and associated endpoints.
- **Prefill/Decode (PD) router**: Coordinates disaggregated prefill and decode workers, merges metadata from prefill outputs with decode worker results, and manages streaming fan-in. PD mode accepts separate policies for prefill and decode worker groups.

#### gRPC Router

The gRPC router provides an industry-first fully Rust implementation of an OpenAI-compatible gRPC inference gateway:

- Streams tokenized requests directly to SRT gRPC workers
- Runs tokenizer, reasoning parser, and tool parser entirely in-process (no Python overhead)
- Supports both single-stage and PD (prefill/decode) worker topologies
- Provides the same `/v1/*` API surface as the HTTP router
- Supports streaming and non-streaming modes for chat completions, generate, embeddings, and classification

The gRPC pipeline stages are:

1. **Dispatch Metadata**: Extract model ID, determine routing configuration
2. **Worker Selection**: Apply load balancing policy to select target worker
3. **Client Acquisition**: Obtain gRPC client connection to selected worker
4. **Request Building**: Tokenize input, apply chat template, build gRPC request
5. **Request Execution**: Send request to worker, handle streaming/non-streaming
6. **Response Processing**: Parse reasoning blocks, extract tool calls, format output

#### OpenAI Router

The OpenAI router proxies requests to external vendors while keeping conversation history and MCP sessions local:

- Preserves headers and SSE streams end-to-end
- Supports `/v1/responses` background jobs with cancellation and deletion
- Manages conversation state at the router tier for privacy compliance
- Handles agentic multi-turn orchestration without persisting data at remote vendor endpoints

### AppContext and Dependency Injection

The `AppContext` is the central shared state container built using a builder pattern. It holds all runtime components:

```rust
pub struct AppContext {
    pub client: Client,                           // HTTP client for worker communication
    pub router_config: RouterConfig,              // Parsed configuration
    pub rate_limiter: Option<Arc<TokenBucket>>,   // Token bucket rate limiter
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub worker_registry: Arc<WorkerRegistry>,     // Thread-safe worker registry
    pub policy_registry: Arc<PolicyRegistry>,     // Per-model policy instances
    pub router_manager: Option<Arc<RouterManager>>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    pub load_monitor: Option<Arc<LoadMonitor>>,
    pub worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,
    pub workflow_engines: Arc<OnceLock<WorkflowEngines>>,
    pub mcp_manager: Arc<OnceLock<Arc<McpManager>>>,
    pub wasm_manager: Option<Arc<WasmModuleManager>>,
    pub worker_service: Arc<WorkerService>,
    pub inflight_tracker: Arc<InFlightRequestTracker>,
}
```

The `AppContextBuilder` provides a fluent API for constructing the context, initializing all components from a `RouterConfig`:

```rust
let app_context = AppContext::from_config(router_config, request_timeout_secs).await?;
```

---

## Installation

### Docker

Pre-built Docker images are available on Docker Hub with multi-architecture support (x86_64 and ARM64):

```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### Prerequisites

- **Rust and Cargo** (for building from source):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  rustc --version
  cargo --version
  ```
- **Python** with `pip` and virtualenv tooling (for Python bindings)

### Building the Rust Binary

```bash
cd sgl-model-gateway
cargo build --release
```

The release profile is optimized for size with full LTO, single codegen unit, and stripped debug symbols:

```toml
[profile.release]
opt-level = "z"       # Optimize for size
lto = "fat"           # Full LTO
codegen-units = 1     # Better optimization, slower compile
strip = true          # Strip debug symbols
```

For CI builds, a separate profile balances compile time and runtime performance:

```toml
[profile.ci]
inherits = "release"
opt-level = 2         # Lighter optimization
lto = "thin"          # Thin LTO balance
codegen-units = 16    # More parallel compilation
```

### Building the Python Package

```bash
pip install maturin

# Fast development mode (debug build, instant iteration)
cd bindings/python
maturin develop

# Production build (optimized wheel with vendored OpenSSL)
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl
```

The Python bindings use abi3 support for Python 3.8+ compatibility.

### Build Caching with sccache

For release builds or CI, you can optionally use sccache to cache compilation artifacts:

```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
cargo build --release
```

Note that sccache and incremental compilation are mutually exclusive. The project defaults to incremental compilation for local development.

---

## Quick Start

### Regular HTTP Routing

```bash
# Using the Rust binary
./target/release/sgl-model-gateway \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware

# Using the Python launcher
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware
```

### gRPC Routing

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

### Prefill/Decode Disaggregation

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --decode http://decode1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

### Multi-Model Inference Gateway

```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url": "http://worker-a:8000", "model_id": "mistral"}'
```

---

## Deployment Modes

### Co-launch Router and Workers

Launch the router and a fleet of SGLang workers in one process. This is the simplest way to get started with data parallelism:

```bash
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

With full router configuration (router-specific flags use the `--router-` prefix):

```bash
python -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode \
  --log-level debug \
  --router-prometheus-port 10001 \
  --router-policy cache_aware
```

The entrypoint is `python -m sglang_router.launch_server` (not `sglang.launch_server`, which is the native/naive DP mode).

### Separate Launch (HTTP)

Run workers independently and point the router at their HTTP endpoints:

```bash
# Worker nodes
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001

# Router node
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

### gRPC Launch

Use SRT gRPC workers for the highest throughput and access to native reasoning/tool pipelines:

```bash
# Workers expose gRPC endpoints
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --grpc-mode \
  --port 20000

# Router connects via gRPC
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

Provide `--tokenizer-path` or `--model-path` (HuggingFace ID or local directory) whenever connection mode resolves to gRPC.

### Prefill-Decode Disaggregation

Split prefill and decode workers for PD-aware caching and balancing:

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --prefill http://prefill2:30002 \
  --decode http://decode1:30011 \
  --decode http://decode2:30012 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

Prefill entries accept an optional bootstrap port (for worker-to-worker communication during PD operations). PD mode merges prefill metadata with decode outputs and streams results back to the client.

### OpenAI Backend Proxy

Proxy OpenAI-compatible endpoints while keeping history and MCP sessions local:

```bash
python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory
```

OpenAI backend mode expects exactly one `--worker-urls` entry per router instance. Load balancing is not applied in this mode.

To route to a custom OpenAI-compatible endpoint:

```bash
python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls http://my-openai-compatible-service:8000 \
  --history-backend postgres
```

### Multi-Model Inference Gateway

Enable IGW mode to route multiple models through a single gateway:

```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://worker-a:8000",
    "model_id": "mistral",
    "priority": 10,
    "labels": {"tier": "gold"}
  }'

curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://worker-b:8000",
    "model_id": "llama3",
    "priority": 20,
    "labels": {"policy": "power_of_two", "tier": "silver"}
  }'
```

IGW mode is automatically enabled when Kubernetes service discovery is turned on:

```
INFO: IGW mode automatically enabled because service discovery is turned on
```

---

## Load Balancing Strategies

SMG supports eight load balancing policies, all implementing the `LoadBalancingPolicy` trait:

```rust
#[async_trait]
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize>;

    fn on_request_complete(&self, _worker_url: &str, _success: bool) {}
    fn name(&self) -> &'static str;
    fn needs_request_text(&self) -> bool { false }
    fn update_loads(&self, _loads: &HashMap<String, isize>) {}
    fn set_mesh_sync(&mut self, _mesh_sync: OptionalMeshSyncManager) {}
    fn reset(&self) {}
    fn as_any(&self) -> &dyn std::any::Any;
}
```

The `SelectWorkerInfo` structure passes routing context to policies:

```rust
pub struct SelectWorkerInfo<'a> {
    pub request_text: Option<&'a str>,     // For cache-aware routing
    pub tokens: Option<&'a [u32]>,         // For prefix-hash routing
    pub headers: Option<&'a HeaderMap>,    // For header-based routing
    pub hash_ring: Option<Arc<HashRing>>,  // Pre-computed for O(log n) lookup
}
```

All policies share a common health-filtering step. The helper function `get_healthy_worker_indices` filters workers that are both healthy and have an open circuit breaker before policy selection begins.

### Random Policy

**CLI flag**: `--policy random`

Uniform random selection among healthy workers. The simplest policy, useful as a baseline or for testing:

```rust
pub struct RandomPolicy;

// Selects a random healthy worker
let mut rng = rand::rng();
let random_idx = rng.random_range(0..healthy_indices.len());
Some(healthy_indices[random_idx])
```

**When to use**: Testing, baseline comparisons, workloads with no cache affinity.

### Round-Robin Policy

**CLI flag**: `--policy round_robin`

Cycles through workers in sequential order using an atomic counter:

```rust
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

// Atomically increment and wrap around
let count = self.counter.fetch_add(1, Ordering::Relaxed);
let selected_idx = count % healthy_indices.len();
```

**When to use**: Simple, predictable distribution when all workers have equal capacity.

### Cache-Aware Policy (Default, Recommended)

**CLI flag**: `--policy cache_aware`

Combines cache affinity with load balancing. This is the default policy and provides the best performance for most workloads.

#### How It Works

The cache-aware policy maintains an approximate radix tree per model for each worker, tracking request history to predict cache locality:

1. **Cache hit path**: When a request's prefix matches a worker's tree with a match rate above `cache_threshold`, the request routes to that worker (high KV cache hit probability)
2. **Cache miss path**: When match rate is below the threshold, routes to the worker with minimum load
3. **Imbalance detection**: When load is imbalanced (both absolute and relative thresholds exceeded), switches to shortest-queue routing while still updating the tree
4. **Background eviction**: Periodically evicts least-recently-used leaf nodes to prevent memory overflow

#### Tuning Parameters

```bash
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.5 \
--eviction-interval 120 \
--max-tree-size 67108864
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cache-threshold` | 0.3 | Minimum prefix match ratio for cache hit (0.0-1.0) |
| `--balance-abs-threshold` | 64 | Absolute load difference to trigger rebalancing |
| `--balance-rel-threshold` | 1.5 | Relative load ratio to trigger rebalancing |
| `--eviction-interval` | 120 | Cache eviction cadence in seconds |
| `--max-tree-size` | 67108864 | Maximum nodes in the approximate radix tree |

#### Imbalance Detection

The system is considered imbalanced when **both** conditions are met:

```
(max_load - min_load) > balance_abs_threshold
AND
max_load > min_load * balance_rel_threshold
```

When imbalanced, the policy routes to the worker with the lowest load (shortest queue) to restore balance, but still updates the tree to maintain cache state for future balanced periods.

#### Mesh Synchronization

The cache-aware policy supports mesh synchronization for multi-node HA deployments. Tree operations (inserts, removals) are synced across mesh nodes so that all replicas maintain consistent cache affinity data:

```rust
// Sync tree operation to mesh if enabled (no-op if mesh is not enabled)
if let Some(ref mesh_sync) = self.mesh_sync {
    let op = TreeOperation::Insert(TreeInsertOp {
        text: text.to_string(),
        tenant: worker_url.to_string(),
    });
    mesh_sync.sync_tree_operation(model_id.to_string(), op)?;
}
```

**When to use**: Most production workloads, especially those with shared prefixes (system prompts, few-shot examples, conversation history).

### Power-of-Two Policy

**CLI flag**: `--policy power_of_two`

Randomly selects two workers and routes to the one with lower load. Provides good load distribution with minimal coordination overhead.

#### How It Works

1. Selects two random healthy workers using an O(1) offset algorithm
2. Compares loads using the highest-fidelity metric available:
   - **Token-based loads** (from LoadMonitor): Used when both workers have cached token data
   - **Request counts** (from Worker.load()): Fallback when either worker is missing token data
3. Selects the worker with lower load

```rust
// O(1) guaranteed-different selection
let idx1 = rng.random_range(0..healthy_indices.len());
let idx2 = (idx1 + 1 + rng.random_range(0..healthy_indices.len() - 1)) % healthy_indices.len();
```

The policy degrades gracefully when metrics are incomplete: if either worker is missing token data, it falls back to request counts for **both** workers to ensure fairness.

**When to use**: Low-latency requirements, workloads without significant cache affinity.

### Bucket Policy

**CLI flag**: `--policy bucket`

Divides workers into load buckets with dynamic boundaries. Workers are assigned to buckets based on their current load, and requests are routed to the least-loaded bucket first.

The bucket boundaries are adjusted periodically in a background thread:

```rust
pub struct BucketConfig {
    pub balance_abs_threshold: usize,        // Default: 32
    pub balance_rel_threshold: f32,          // Default: 1.0001
    pub bucket_adjust_interval_secs: usize,  // Default: 5
}
```

**When to use**: When you need more granular load grouping than power-of-two but less overhead than cache-aware.

### Manual (Sticky Session) Policy

**CLI flag**: `--policy manual`

Provides sticky session routing where each unique routing key is consistently mapped to the same worker. Unlike consistent hashing, this policy does **not** redistribute sessions when workers are added.

#### Configuration

```bash
--policy manual \
--max-idle-secs 14400 \
--assignment-mode random
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-idle-secs` | 14400 (4 hours) | Maximum idle time before session eviction |
| `--assignment-mode` | random | Assignment mode for new routing keys |

Assignment modes:

| Mode | Behavior |
|------|----------|
| `random` | Random worker selection for new keys |
| `min_load` | Select worker with minimum running requests |
| `min_group` | Select worker with fewest active routing keys |

#### Usage

Clients must send an `X-SMG-Routing-Key` header for sticky session routing:

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "X-SMG-Routing-Key: user-session-123" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [...]}'
```

The policy maintains up to 2 candidate workers per routing key for fast failover when the primary worker becomes unhealthy.

**When to use**: Stateful chat sessions, conversational AI with worker-local context.

### Prefix Hash Policy

**CLI flag**: `--policy prefix_hash`

A lightweight alternative to the full radix tree cache_aware policy. Routes requests based on a hash of their prefix tokens using a consistent hash ring.

#### Algorithm

1. Extract first N tokens from the request (configurable prefix length)
2. Hash the token sequence using xxhash for fast, stable hashing
3. Use consistent hash ring to find the target worker (O(log n) binary search)
4. If worker is overloaded (load > avg * load_factor), walk clockwise to the next worker

#### Configuration

```bash
--policy prefix_hash \
--prefix-token-count 256 \
--prefix-hash-load-factor 1.25
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prefix-token-count` | 256 | Number of prefix tokens to hash |
| `--prefix-hash-load-factor` | 1.25 | Load factor threshold for ring walking |

#### Comparison with Cache-Aware

| Aspect | prefix_hash | cache_aware (radix) |
|--------|-------------|---------------------|
| Lookup | O(log n) | O(prefix_len) |
| Memory | O(workers x vn) | O(total_tokens) |
| Update | O(1) | O(prefix_len) |
| Precision | Prefix grouping | Exact matching |

Prefix hash trades optimal cache utilization for predictable O(log n) performance.

**When to use**: High-throughput workloads where predictable latency matters more than optimal cache hit rates.

### Consistent Hashing Policy

Available as a routing policy option, consistent hashing uses a hash ring with 150 virtual nodes per worker for even distribution. It provides O(log n) lookup with minimal redistribution (approximately 1/N keys) when topology changes.

Supports two header-based routing modes:
- `X-SMG-Target-Worker`: Direct routing to a specific worker by URL
- `X-SMG-Routing-Key`: Consistent hash routing for session affinity

### Policy Selection Guide

| Scenario | Recommended Policy | Reason |
|----------|-------------------|--------|
| General production | `cache_aware` | Best balance of cache locality and load distribution |
| Shared prefixes (system prompts) | `cache_aware` | Maximizes KV cache reuse (~92% throughput improvement) |
| Diverse, uncorrelated requests | `power_of_two` | Good distribution with low overhead |
| Stateful chat sessions | `manual` | Sticky sessions with fast failover |
| High-throughput, latency-sensitive | `prefix_hash` | Predictable O(log n) performance |
| Simple testing | `random` or `round_robin` | Minimal configuration |
| PD decode workers | `power_of_two` | Low-latency selection for decode stage |

---

## Data Parallelism Routing

### Native DP vs SMG-Based DP

SGLang supports two data parallelism approaches:

| Feature | Native DP | SMG-Based DP |
|---------|-----------|--------------|
| Entrypoint | `sglang.launch_server` | `sglang_router.launch_server` |
| Load Balancing | In-process (round_robin, total_requests, total_tokens) | Advanced policies (cache_aware, power_of_two, etc.) |
| Cache Awareness | No | Yes - significant cache hit improvement |
| Multi-Node | Limited | Full support |
| Health Monitoring | Basic | Circuit breakers, health checks |
| Observability | Basic | 40+ Prometheus metrics, OpenTelemetry |
| Hot Worker Add/Remove | No | Yes |
| RL Rollout Support | Limited | Full support |

**Native DP is not recommended for production use.** Always use SMG-based DP for production deployments.

### SMG-Based DP Deployment

#### Option A: Co-launch (Simplest)

```bash
python -m sglang_router.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

#### Option B: Separate Launch (Multi-Node)

```bash
# Node 1: Worker
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000

# Node 2: Worker
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000

# Router node
python -m sglang_router.launch_router \
  --worker-urls http://node1:8000 http://node2:8000 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

#### Option C: Dynamic Registration (Elastic)

```bash
# Launch SMG first
python -m sglang_router.launch_router \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url": "http://worker1:8000"}'

curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url": "http://worker2:8000"}'
```

### DP-Aware Scheduling

Enable DP-aware scheduling with `--dp-aware` to optimize routing decisions based on data parallelism topology:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --dp-aware \
  --policy cache_aware
```

### Data Parallelism Attention (DPA)

For models using Multi-Head Latent Attention (MLA) such as DeepSeek, MiniMax, and Kimi-K2, DPA applies data parallelism specifically to the attention component:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm
```

DPA can be combined with SMG for the best production experience.

### Verifying Traffic Distribution

```bash
# Check worker status
curl http://localhost:30000/workers

# Check load distribution
curl http://localhost:30000/get_loads

# Monitor via Prometheus
smg_router_requests_total{model="..."}
smg_worker_requests_active{worker="..."}
```

---

## Health Checking

### Architecture

The health checker runs as a background task spawned by the `WorkerRegistry`. It probes each worker at a configurable interval and updates worker health status based on success/failure thresholds.

### Configuration

```bash
--health-check-interval-secs 30 \
--health-check-timeout-secs 5 \
--health-failure-threshold 3 \
--health-success-threshold 2 \
--health-check-endpoint /health
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--health-check-interval-secs` | 60 | Interval between health check probes |
| `--health-check-timeout-secs` | 5 | Timeout for each health check request |
| `--health-failure-threshold` | 3 | Consecutive failures before marking unhealthy |
| `--health-success-threshold` | 2 | Consecutive successes before marking healthy |
| `--health-check-endpoint` | /health | HTTP endpoint to probe |
| `--disable-health-check` | false | Disable all health checks |

### Health Check Flow

1. Every `check_interval_secs`, the health checker iterates over all registered workers
2. For each worker, it sends an HTTP GET request to `health_check_endpoint`
3. If the response status is 2xx, the success counter increments
4. If the request fails or times out, the failure counter increments
5. When consecutive failures reach `failure_threshold`, the worker is marked unhealthy
6. When consecutive successes reach `success_threshold`, the worker is marked healthy again

### Readiness vs Liveness

The gateway provides distinct endpoints for Kubernetes probes:

- **`/liveness`**: Always returns 200 OK. Use for liveness probes (is the gateway process alive?)
- **`/readiness`**: Returns 200 only when healthy workers are available. Returns 503 when no healthy workers exist. Use for readiness probes (is the gateway ready to serve traffic?)

Readiness checks are mode-aware:

```rust
// PD mode requires both prefill AND decode workers
let has_prefill = healthy_workers.iter().any(|w| matches!(w.worker_type(), WorkerType::Prefill));
let has_decode = healthy_workers.iter().any(|w| matches!(w.worker_type(), WorkerType::Decode));
let is_ready = has_prefill && has_decode;

// Regular mode requires at least one healthy worker
let is_ready = !healthy_workers.is_empty();
```

---

## Reliability and Flow Control

### Retries

Configure exponential backoff retries for transient failures:

```bash
--retry-max-retries 5 \
--retry-initial-backoff-ms 50 \
--retry-max-backoff-ms 30000 \
--retry-backoff-multiplier 1.5 \
--retry-jitter-factor 0.2
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--retry-max-retries` | 5 | Maximum retry attempts |
| `--retry-initial-backoff-ms` | 50 | Initial backoff duration (ms) |
| `--retry-max-backoff-ms` | 30000 | Maximum backoff duration (ms) |
| `--retry-backoff-multiplier` | 1.5 | Exponential backoff multiplier |
| `--retry-jitter-factor` | 0.2 | Random jitter factor (0.0-1.0) |
| `--disable-retries` | false | Disable retries entirely |

**Retryable status codes**: 408 (Request Timeout), 429 (Too Many Requests), 500 (Internal Server Error), 502 (Bad Gateway), 503 (Service Unavailable), 504 (Gateway Timeout).

The backoff formula is:

```
delay_ms = min(initial_backoff_ms * multiplier^attempt, max_backoff_ms)
adjusted_delay = delay_ms * (1 + random(-jitter, +jitter))
```

When `--disable-retries` is set, the effective `max_retries` is overridden to 1 (single attempt, no retries).

### Circuit Breaker

Per-worker circuit breakers prevent cascading failures by tracking consecutive failures and opening the circuit when a worker is unreliable.

#### States

```
    ┌──────────┐
    │  Closed  │ ◄─── Normal operation, requests allowed
    │(healthy) │
    └────┬─────┘
         │ failure_threshold consecutive failures
         ▼
    ┌──────────┐
    │   Open   │ ◄─── Failing, requests rejected immediately
    │(failing) │
    └────┬─────┘
         │ timeout_duration_secs elapsed
         ▼
    ┌──────────┐
    │ Half-Open│ ◄─── Testing recovery, limited requests allowed
    │(testing) │
    └────┬─────┘
         │ success_threshold consecutive successes
         ▼
    ┌──────────┐
    │  Closed  │
    └──────────┘
```

#### Configuration

```bash
--cb-failure-threshold 10 \
--cb-success-threshold 3 \
--cb-timeout-duration-secs 60 \
--cb-window-duration-secs 120 \
--disable-circuit-breaker false
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cb-failure-threshold` | 10 | Consecutive failures to open circuit |
| `--cb-success-threshold` | 3 | Successes needed to close from half-open |
| `--cb-timeout-duration-secs` | 60 | Time before half-open attempt |
| `--cb-window-duration-secs` | 120 | Sliding window for failure counting |
| `--disable-circuit-breaker` | false | Disable circuit breaker |

When `--disable-circuit-breaker` is set, `failure_threshold` is overridden to `u32::MAX`, effectively preventing the circuit from ever opening.

Workers with an open circuit breaker are excluded from policy selection via the `get_healthy_worker_indices` helper.

### Rate Limiting and Queuing

The gateway implements token-bucket rate limiting with a FIFO request queue:

```bash
--max-concurrent-requests 256 \
--rate-limit-tokens-per-second 512 \
--queue-size 128 \
--queue-timeout-secs 30
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-concurrent-requests` | -1 (disabled) | Maximum concurrent requests (-1 disables) |
| `--rate-limit-tokens-per-second` | (same as max) | Token bucket refill rate |
| `--queue-size` | 100 | Queue capacity for pending requests |
| `--queue-timeout-secs` | 60 | Maximum queue wait time |

Response codes:
- **429 Too Many Requests**: When the queue is full
- **408 Request Timeout**: When queue timeout expires before the request can be processed

When `max_concurrent_requests` is -1 (default), rate limiting is completely disabled and all requests proceed immediately.

The concurrency limiter middleware runs as part of the Axum middleware stack:

```rust
.route_layer(axum::middleware::from_fn_with_state(
    app_state.clone(),
    middleware::concurrency_limit_middleware,
))
```

---

## API Compatibility

### Inference Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | SGLang generate API |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming/tool calls) |
| `POST` | `/v1/completions` | OpenAI-compatible text completions |
| `POST` | `/v1/embeddings` | Embedding generation (HTTP and gRPC) |
| `POST` | `/v1/rerank` | Reranking requests |
| `POST` | `/rerank` | Reranking (alternative path) |
| `POST` | `/v1/classify` | Text classification |

### Tokenization Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/tokenize` | Tokenize text to token IDs (single or batch) |
| `POST` | `/v1/detokenize` | Convert token IDs back to text |
| `POST` | `/v1/tokenizers` | Register a new tokenizer (async) |
| `GET` | `/v1/tokenizers` | List all registered tokenizers |
| `GET` | `/v1/tokenizers/{id}` | Get tokenizer info by UUID |
| `GET` | `/v1/tokenizers/{id}/status` | Check async loading status |
| `DELETE` | `/v1/tokenizers/{id}` | Remove a tokenizer |

### Response and Conversation APIs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/responses` | Create background responses (agentic loops) |
| `GET` | `/v1/responses/{id}` | Retrieve stored response |
| `POST` | `/v1/responses/{id}/cancel` | Cancel background response |
| `DELETE` | `/v1/responses/{id}` | Delete response |
| `GET` | `/v1/responses/{id}/input_items` | List response input items |
| `POST` | `/v1/conversations` | Create conversation |
| `GET` | `/v1/conversations/{id}` | Get conversation |
| `POST` | `/v1/conversations/{id}` | Update conversation |
| `DELETE` | `/v1/conversations/{id}` | Delete conversation |
| `GET` | `/v1/conversations/{id}/items` | List conversation items |
| `POST` | `/v1/conversations/{id}/items` | Add items |
| `GET` | `/v1/conversations/{id}/items/{item_id}` | Get specific item |
| `DELETE` | `/v1/conversations/{id}/items/{item_id}` | Delete item |

### Parser Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/parse/reasoning` | Separate reasoning content from normal text |
| `POST` | `/parse/function_call` | Parse function/tool calls from text |

### Worker Management APIs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workers` | Queue worker registration (returns 202 Accepted) |
| `GET` | `/workers` | List workers with health, load, and policy metadata |
| `GET` | `/workers/{worker_id}` | Inspect specific worker or job queue entry |
| `PUT` | `/workers/{worker_id}` | Queue worker update |
| `DELETE` | `/workers/{worker_id}` | Queue worker removal |

### Admin and Health Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/liveness` | Health check (always returns OK) |
| `GET` | `/readiness` | Readiness check (checks healthy worker availability) |
| `GET` | `/health` | Alias for liveness |
| `GET` | `/health_generate` | Health generate test |
| `GET` | `/engine_metrics` | Engine-level metrics from workers |
| `GET` | `/v1/models` | List available models |
| `GET` | `/get_model_info` | Get model information |
| `GET` | `/server_info` | Get server information |
| `POST` | `/flush_cache` | Clear all caches |
| `GET` | `/get_loads` | Get all worker loads |
| `POST` | `/wasm` | Upload WASM module |
| `GET` | `/wasm` | List WASM modules |
| `DELETE` | `/wasm/{module_uuid}` | Remove WASM module |

### HA (Mesh) Management APIs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ha/status` | Get cluster status |
| `GET` | `/ha/health` | Get mesh health |
| `GET` | `/ha/workers` | Get worker states across mesh |
| `GET` | `/ha/workers/{worker_id}` | Get specific worker state |
| `GET` | `/ha/policies` | Get policy states |
| `GET` | `/ha/policies/{model_id}` | Get specific model policy state |
| `GET` | `/ha/config/{key}` | Get app config |
| `POST` | `/ha/config` | Update app config |
| `POST` | `/ha/rate-limit` | Set global rate limit |
| `GET` | `/ha/rate-limit` | Get global rate limit |
| `GET` | `/ha/rate-limit/stats` | Get rate limit statistics |
| `POST` | `/ha/shutdown` | Trigger graceful shutdown |

---

## Tokenizer Management

### Tokenizer Sources

The gateway supports multiple tokenizer backends:

- **HuggingFace**: Load from HuggingFace Hub by model ID
- **Local**: Load from local `tokenizer.json` or directory
- **Tiktoken**: Auto-detect OpenAI GPT models

### Configuration

```bash
# HuggingFace model (loads tokenizer from Hub)
--model-path meta-llama/Llama-3.1-8B-Instruct

# Local tokenizer file
--tokenizer-path /path/to/tokenizer.json

# Chat template override
--chat-template /path/to/template.jinja
```

### Two-Level Tokenizer Caching

| Cache Level | Type | Description |
|-------------|------|-------------|
| L0 | Exact match | Whole-string caching for repeated prompts |
| L1 | Prefix match | Prefix boundary matching for incremental prompts |

```bash
--enable-l0-cache \
--l0-max-entries 10000 \
--enable-l1-cache \
--l1-max-memory 52428800  # 50MB
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable-l0-cache` | false | Enable L0 exact match cache |
| `--l0-max-entries` | 10,000 | Maximum L0 cache entries |
| `--enable-l1-cache` | false | Enable L1 prefix match cache |
| `--l1-max-memory` | 50MB | Maximum L1 cache memory in bytes |

### Dynamic Tokenizer Registration

Tokenizers are loaded via the workflow engine (`TokenizerRegistration` step) and can be registered at startup, when workers connect, or via the API:

```bash
# Register from HuggingFace
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'

# Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Tokenizer registration queued"
}

# Check loading status
curl http://localhost:30000/v1/tokenizers/550e8400-.../status
```

---

## Reasoning Parser Integration

The gateway includes built-in reasoning parsers for models that use Chain-of-Thought (CoT) reasoning with explicit thinking blocks.

### Supported Parsers

| Parser ID | Model Family | Think Tokens |
|-----------|--------------|--------------|
| `deepseek-r1` | DeepSeek-R1 | `<thinkinfo>...</thinkinfo>` (initial reasoning) |
| `qwen3` | Qwen-3 | `<thinkinfo>...</thinkinfo>` |
| `qwen3-thinking` | Qwen-3 Thinking | `<thinkinfo>...</thinkinfo>` (initial reasoning) |
| `kimi` | Kimi K2 | Unicode think tokens |
| `glm45` | GLM-4.5/4.6/4.7 | `<thinkinfo>...</thinkinfo>` |
| `step3` | Step-3 | `<thinkinfo>...</thinkinfo>` |
| `minimax` | MiniMax | `<thinkinfo>...</thinkinfo>` |

### Usage

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek-r1
```

The gRPC router automatically:

1. Detects reasoning blocks in streaming output
2. Separates reasoning content from normal text
3. Applies incremental streaming parsing with buffer management
4. Handles partial token detection for correct streaming behavior

---

## Tool Call Parsing

The gateway supports parsing function/tool calls from LLM outputs in multiple formats.

### Supported Formats

| Parser | Format | Description |
|--------|--------|-------------|
| `json` | JSON | Standard JSON tool calls |
| `python` | Pythonic | Python function call syntax |
| `xml` | XML | XML-formatted tool calls |

### Usage

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tool-call-parser json
```

---

## MCP Integration

The gateway provides native Model Context Protocol (MCP) client integration for tool execution.

### Supported Transports

| Transport | Description |
|-----------|-------------|
| STDIO | Local process execution |
| SSE | Server-Sent Events (HTTP) |
| Streamable | Bidirectional streaming |

### Configuration File

```yaml
servers:
  - name: "filesystem"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    protocol: "stdio"
    required: false

  - name: "github"
    url: "https://api.github.com/mcp"
    token: "ghp_xxxxx"
    protocol: "sse"
    required: false

  - name: "custom-tools"
    url: "https://tools.example.com/mcp"
    protocol: "streamable"
    required: true

pool:
  max_connections: 100
  idle_timeout: 300

proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8443"
  no_proxy: "localhost,127.0.0.1,*.internal"

inventory:
  enable_refresh: true
  tool_ttl: 300
  refresh_interval: 300
```

### Usage

```bash
python -m sglang_router.launch_router \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000
```

MCP servers are registered via the workflow engine with retry logic (100 attempts, 2-hour timeout for STDIO servers). The tool inventory is cached with configurable TTL and periodic background refresh (default: every 10 minutes).

---

## Service Discovery (Kubernetes)

Enable automatic worker discovery via Kubernetes pod selectors:

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

When service discovery is enabled, IGW mode is automatically activated.

### PD Mode Discovery

```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```

Prefill pods can expose bootstrap ports via the `sglang.ai/bootstrap-port` annotation.

### RBAC Requirements

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-gateway
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-gateway
  namespace: production
subjects:
- kind: ServiceAccount
  name: sglang-gateway
  namespace: production
roleRef:
  kind: Role
  name: sglang-gateway
  apiGroup: rbac.authorization.k8s.io
```

---

## History and Data Connectors

| Backend | CLI Flag | Description |
|---------|----------|-------------|
| Memory | `--history-backend memory` | In-memory storage (default) |
| None | `--history-backend none` | No persistence |
| Oracle | `--history-backend oracle` | Oracle Autonomous Database |
| PostgreSQL | `--history-backend postgres` | PostgreSQL Database |
| Redis | `--history-backend redis` | Redis |

### Oracle Configuration

```bash
export ATP_DSN="(description=(address=(protocol=tcps)(port=1522)(host=adb.region.oraclecloud.com))(connect_data=(service_name=service_name)))"
export ATP_USER="admin"
export ATP_PASSWORD="secret"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle
```

### PostgreSQL Configuration

```bash
export POSTGRES_DB_URL="postgres://user:password@host:5432/dbname"

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend postgres
```

### Redis Configuration

```bash
export REDIS_URL="redis://localhost:6379"
export REDIS_POOL_MAX=16

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend redis \
  --redis-retention-days 30
```

Use `--redis-retention-days -1` for persistent storage (no expiration).

---

## WASM Middleware

The gateway supports WebAssembly (WASM) middleware modules for custom request/response processing.

### Overview

| Attach Point | When Executed | Use Cases |
|--------------|---------------|-----------|
| `OnRequest` | Before forwarding to workers | Auth, rate limiting, request modification |
| `OnResponse` | After receiving worker response | Logging, response modification, error handling |

| Action | Description |
|--------|-------------|
| `Continue` | Proceed without modification |
| `Reject(status)` | Reject request with HTTP status code |
| `Modify(...)` | Modify headers, body, or status |

### Building Modules

```bash
rustup target add wasm32-wasip2
cargo install wasm-tools

cargo build --target wasm32-wasip2 --release
wasm-tools component new \
  target/wasm32-wasip2/release/my_middleware.wasm \
  -o my_middleware.component.wasm
```

### Deploying

```bash
# Enable WASM
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-wasm

# Upload module
curl -X POST http://localhost:30000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "auth-middleware",
      "file_path": "/path/to/auth.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'
```

---

## Language Bindings

### Python Bindings

PyO3-based wrapper around the Rust gateway library. Install via:

```bash
pip install sglang-router
# or
pip install "sglang[all]"
```

Key components:
- `RouterArgs` dataclass with 50+ configuration options
- `Router.from_args()` for programmatic startup
- CLI commands: `smg launch`, `smg server`, `python -m sglang_router.launch_router`

### Go Bindings

High-performance gRPC client library for Go-based infrastructure:

```
┌─────────────────────────────────────────┐
│         High-Level Go API               │
│   (client.go - OpenAI-style interface)  │
├─────────────────────────────────────────┤
│         gRPC Layer                      │
├─────────────────────────────────────────┤
│         Rust FFI Layer                  │
│   (Tokenization, Parsing, Conversion)   │
└─────────────────────────────────────────┘
```

---

## Security and Authentication

### Router API Key

```bash
python -m sglang_router.launch_router \
  --api-key "your-router-api-key" \
  --worker-urls http://worker1:8000
```

### TLS (HTTPS) for Gateway Server

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --tls-cert-path /path/to/server.crt \
  --tls-key-path /path/to/server.key
```

Uses rustls with the ring crypto provider for TLS termination.

### mTLS for Worker Communication

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

Multiple CA certificates can be added with multiple `--ca-cert-path` flags. TCP keepalive (30 seconds) is enabled for long-lived connections.

### Control Plane Authentication

The gateway supports role-based access control (RBAC) for control plane APIs with two authentication methods:

#### API Key Authentication

```bash
python -m sglang_router.launch_router \
  --control-plane-api-keys 'svc1:CI Pipeline:admin:secret-key-123' \
                           'svc2:Monitoring:user:readonly-key-456'
```

Format: `id:name:role:key` where role is `admin` or `user`.

#### JWT/OIDC Authentication

```bash
python -m sglang_router.launch_router \
  --jwt-issuer "https://login.microsoftonline.com/{tenant-id}/v2.0" \
  --jwt-audience "api://my-gateway-client-id" \
  --jwt-role-mapping 'Gateway.Admins=admin' 'Gateway.Users=user'
```

### Full Security Configuration

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key" \
  --control-plane-api-keys 'admin:Admin Service:admin:admin-secret'
```

---

## Production Deployment

### High Availability Architecture

```
                ┌─────────────────┐
                │  Load Balancer  │
                │   (L4/L7)       │
                └────────┬────────┘
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
    │  Gateway  │  │  Gateway  │  │  Gateway  │
    │ Replica 1 │  │ Replica 2 │  │ Replica 3 │
    └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
    │  Worker   │  │  Worker   │  │  Worker   │
    │  Pod 1    │  │  Pod 2    │  │  Pod N    │
    └───────────┘  └───────────┘  └───────────┘
```

### HA Trade-offs

| Component | Shared Across Replicas | Impact |
|-----------|----------------------|--------|
| Worker Registry | No (independent) | Each replica discovers workers independently |
| Radix Cache Tree | No (independent) | Cache hits may decrease by 10-20% |
| Circuit Breaker State | No (independent) | Each replica tracks failures independently |
| Rate Limiting | No (independent) | Limits apply per-replica, not globally |

### Mesh Mode for HA

Enable mesh mode for cross-replica state synchronization:

```bash
python -m sglang_router.launch_router \
  --enable-mesh \
  --mesh-server-name gateway-1 \
  --mesh-host 0.0.0.0 \
  --mesh-port 39527 \
  --mesh-peer-urls 10.0.0.2:39527 \
  --enable-igw \
  --service-discovery \
  --selector app=sglang-worker
```

Mesh mode synchronizes:
- Worker registry state across nodes
- Policy (cache tree) state across nodes
- Rate limit windows for consistent enforcement

### Kubernetes Deployment

#### Worker Pod Labeling

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-worker
  namespace: production
spec:
  replicas: 4
  selector:
    matchLabels:
      app: sglang-worker
      component: inference
  template:
    metadata:
      labels:
        app: sglang-worker
        component: inference
        model: llama-3-8b
    spec:
      containers:
      - name: worker
        image: lmsysorg/sglang:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 20000
          name: grpc
```

#### PD Mode Labeling

```yaml
# Prefill Worker
metadata:
  labels:
    app: sglang-worker
    component: prefill
  annotations:
    sglang.ai/bootstrap-port: "9001"

# Decode Worker
metadata:
  labels:
    app: sglang-worker
    component: decode
```

### Performance Tuning Recommendations

| Parameter | Recommendation | Reason |
|-----------|---------------|--------|
| `--policy` | `cache_aware` | Best for repeated prompts, ~92% throughput improvement |
| `--max-concurrent-requests` | 2-4x worker count | Prevent overload while maximizing throughput |
| `--queue-size` | 2x max-concurrent | Buffer for burst traffic |
| `--request-timeout-secs` | Based on max generation length | Prevent stuck requests |
| Connection mode | gRPC | Highest throughput with native Rust tokenization |

---

## Configuration Reference

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--host` | str | 0.0.0.0 | Router bind address |
| `--port` | int | 30000 | Router bind port |
| `--worker-urls` | list | [] | Worker URLs (HTTP or gRPC) |
| `--policy` | str | cache_aware | Routing policy |
| `--max-concurrent-requests` | int | -1 | Concurrency limit (-1 disables) |
| `--request-timeout-secs` | int | 1800 | Request timeout |
| `--max-payload-size` | int | 536870912 | Maximum request payload (512MB) |
| `--shutdown-grace-period-secs` | int | 180 | Grace period for in-flight requests |
| `--backend` | enum | sglang | Backend runtime (sglang, vllm, trtllm, openai, anthropic) |
| `--dp-aware` | flag | false | Enable DP-aware scheduling |
| `--enable-igw` | flag | false | Enable Inference Gateway mode |

### Routing Policy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cache-threshold` | f32 | 0.3 | Cache match ratio for cache_aware |
| `--balance-abs-threshold` | int | 64 | Absolute load diff for rebalancing |
| `--balance-rel-threshold` | f32 | 1.5 | Relative load ratio for rebalancing |
| `--eviction-interval` | int | 120 | Cache eviction interval (seconds) |
| `--max-tree-size` | int | 67108864 | Max radix tree nodes |
| `--prefix-token-count` | int | 256 | Prefix tokens for prefix_hash |
| `--prefix-hash-load-factor` | f64 | 1.25 | Load factor for prefix_hash |
| `--max-idle-secs` | int | 14400 | Max idle time for manual policy |
| `--assignment-mode` | str | random | Assignment mode (random/min_load/min_group) |

### Prefill/Decode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pd-disaggregation` | flag | false | Enable PD mode |
| `--prefill` | list | [] | Prefill URLs + optional bootstrap ports |
| `--decode` | list | [] | Decode URLs |
| `--prefill-policy` | str | None | Override policy for prefill nodes |
| `--decode-policy` | str | None | Override policy for decode nodes |
| `--worker-startup-timeout-secs` | int | 1800 | Worker init timeout |
| `--worker-startup-check-interval` | int | 30 | Worker startup check interval |

### Kubernetes Discovery

| Parameter | Type | Description |
|-----------|------|-------------|
| `--service-discovery` | flag | Enable Kubernetes discovery |
| `--selector` | list | Label selectors (key=value) |
| `--prefill-selector` | list | PD mode prefill selectors |
| `--decode-selector` | list | PD mode decode selectors |
| `--service-discovery-namespace` | str | Namespace to watch |
| `--service-discovery-port` | int | Worker port (default 80) |

### Retry Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--retry-max-retries` | int | 5 | Maximum retry attempts |
| `--retry-initial-backoff-ms` | int | 50 | Initial backoff (ms) |
| `--retry-max-backoff-ms` | int | 30000 | Maximum backoff (ms) |
| `--retry-backoff-multiplier` | f32 | 1.5 | Exponential multiplier |
| `--retry-jitter-factor` | f32 | 0.2 | Jitter factor (0.0-1.0) |
| `--disable-retries` | flag | false | Disable retries |

### Circuit Breaker

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cb-failure-threshold` | int | 10 | Failures to open circuit |
| `--cb-success-threshold` | int | 3 | Successes to close circuit |
| `--cb-timeout-duration-secs` | int | 60 | Half-open timeout |
| `--cb-window-duration-secs` | int | 120 | Failure counting window |
| `--disable-circuit-breaker` | flag | false | Disable circuit breaker |

### Health Checks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--health-failure-threshold` | int | 3 | Failures before unhealthy |
| `--health-success-threshold` | int | 2 | Successes before healthy |
| `--health-check-timeout-secs` | int | 5 | Probe timeout |
| `--health-check-interval-secs` | int | 60 | Probe interval |
| `--health-check-endpoint` | str | /health | Probe endpoint |
| `--disable-health-check` | flag | false | Disable health checks |

### Tokenizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-path` | str | None | HuggingFace model ID or local path |
| `--tokenizer-path` | str | None | Explicit tokenizer path (overrides model-path) |
| `--chat-template` | str | None | Chat template path override |
| `--enable-l0-cache` | flag | false | Enable L0 exact match cache |
| `--l0-max-entries` | int | 10000 | L0 cache max entries |
| `--enable-l1-cache` | flag | false | Enable L1 prefix match cache |
| `--l1-max-memory` | int | 52428800 | L1 cache max memory (50MB) |

### Parsers

| Parameter | Type | Description |
|-----------|------|-------------|
| `--reasoning-parser` | str | Parser for reasoning models (deepseek-r1, qwen3, etc.) |
| `--tool-call-parser` | str | Parser for tool calls (json, python, xml) |
| `--mcp-config-path` | str | Path to MCP configuration YAML |

### TLS/mTLS

| Parameter | Type | Description |
|-----------|------|-------------|
| `--tls-cert-path` | str | Server certificate for HTTPS (PEM) |
| `--tls-key-path` | str | Server private key for HTTPS (PEM) |
| `--client-cert-path` | str | Client certificate for worker mTLS (PEM) |
| `--client-key-path` | str | Client private key for worker mTLS (PEM) |
| `--ca-cert-path` | str | CA certificate for worker verification (PEM, repeatable) |

### Observability

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--prometheus-host` | str | 0.0.0.0 | Prometheus metrics host |
| `--prometheus-port` | int | 29000 | Prometheus metrics port |
| `--prometheus-duration-buckets` | list | (default set) | Custom duration histogram buckets |
| `--enable-trace` | flag | false | Enable OpenTelemetry tracing |
| `--otlp-traces-endpoint` | str | localhost:4317 | OTLP collector endpoint |
| `--log-level` | str | info | Log level (debug/info/warn/error) |
| `--log-dir` | str | None | Log file directory |
| `--json-log` | flag | false | Structured JSON log output |
| `--request-id-headers` | list | (default set) | Custom request ID headers |
| `--cors-allowed-origins` | list | [] | CORS allowed origins |

### Authentication

| Parameter | Type | Description |
|-----------|------|-------------|
| `--api-key` | str | Router API key for client authentication |
| `--control-plane-api-keys` | list | Control plane API keys (format: id:name:role:key) |
| `--jwt-issuer` | str | JWT issuer URL for OIDC |
| `--jwt-audience` | str | Expected JWT audience |
| `--jwt-jwks-uri` | str | Explicit JWKS URI |
| `--jwt-role-claim` | str | JWT claim name for role (default: roles) |
| `--jwt-role-mapping` | list | Role mapping (format: idp_role=gateway_role) |
| `--disable-audit-logging` | flag | Disable control plane audit logging |

### Rate Limiting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max-concurrent-requests` | int | -1 | Max concurrent requests (-1 disables) |
| `--queue-size` | int | 100 | Pending request queue size |
| `--queue-timeout-secs` | int | 60 | Queue wait timeout |
| `--rate-limit-tokens-per-second` | int | (same as max) | Token bucket refill rate |

### History Backend

| Parameter | Type | Description |
|-----------|------|-------------|
| `--history-backend` | str | Storage backend (memory/none/oracle/postgres/redis) |

### Mesh (HA)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable-mesh` | flag | false | Enable mesh mode for HA |
| `--mesh-server-name` | str | (random) | Mesh node name |
| `--mesh-host` | str | 0.0.0.0 | Mesh bind host |
| `--mesh-port` | int | 39527 | Mesh bind port |
| `--mesh-peer-urls` | list | [] | Peer gateway addresses |

### WASM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable-wasm` | flag | false | Enable WASM middleware support |

---

## Monitoring and Observability

### Prometheus Metrics

Enable with `--prometheus-host`/`--prometheus-port` (defaults to `0.0.0.0:29000`).

#### Metric Categories (40+ metrics)

| Layer | Prefix | Key Metrics |
|-------|--------|-------------|
| HTTP | `smg_http_*` | `requests_total`, `request_duration_seconds`, `responses_total`, `connections_active`, `rate_limit_total` |
| Router | `smg_router_*` | `requests_total`, `request_duration_seconds`, `request_errors_total`, `stage_duration_seconds`, `upstream_responses_total` |
| Inference | `smg_router_*` | `ttft_seconds`, `tpot_seconds`, `tokens_total`, `generation_duration_seconds` |
| Worker | `smg_worker_*` | `pool_size`, `connections_active`, `requests_active`, `health_checks_total`, `selection_total`, `errors_total` |
| Circuit Breaker | `smg_worker_cb_*` | `state`, `transitions_total`, `outcomes_total`, `consecutive_failures` |
| Retry | `smg_worker_*` | `retries_total`, `retries_exhausted_total`, `retry_backoff_seconds` |
| Discovery | `smg_discovery_*` | `registrations_total`, `deregistrations_total`, `sync_duration_seconds` |
| MCP | `smg_mcp_*` | `tool_calls_total`, `tool_duration_seconds`, `servers_active` |
| Database | `smg_db_*` | `operations_total`, `operation_duration_seconds`, `connections_active` |

#### Duration Histogram Buckets

```
1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 15s, 30s, 45s, 60s, 90s, 120s, 180s, 240s
```

### Essential PromQL Queries

#### Request Rate and Latency

```promql
# Request rate by endpoint
sum(rate(smg_http_requests_total[5m])) by (path, method)

# P50 latency
histogram_quantile(0.50, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# P99 latency
histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(smg_http_responses_total{status=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))
```

#### Worker Health

```promql
# Healthy workers
sum(smg_worker_pool_size)

# Active connections per worker
smg_worker_connections_active

# Worker health check failures
sum(rate(smg_worker_health_checks_total{result="failure"}[5m])) by (worker_id)
```

#### Circuit Breaker Status

```promql
# Circuit breaker states (0=closed, 1=open, 2=half-open)
smg_worker_cb_state

# Workers with open circuits
count(smg_worker_cb_state == 1)
```

#### Inference Performance (gRPC mode)

```promql
# Time to first token (P50)
histogram_quantile(0.50, sum(rate(smg_router_ttft_seconds_bucket[5m])) by (le, model))

# Time per output token (P99)
histogram_quantile(0.99, sum(rate(smg_router_tpot_seconds_bucket[5m])) by (le, model))

# Token throughput
sum(rate(smg_router_tokens_total[5m])) by (model, direction)
```

### OpenTelemetry Tracing

Enable distributed tracing with OTLP export:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

Features:
- OTLP/gRPC exporter (default port 4317)
- W3C Trace Context propagation for HTTP and gRPC
- Batch span processing (500ms delay, 64 span batch size)
- Custom filtering to reduce noise
- Trace context injection into upstream worker requests
- Service name: `sgl-router`

### Logging

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs \
  --json-log
```

Structured tracing through the `tracing` crate with optional file sink. Log levels: `debug`, `info`, `warn`, `error`. JSON format available for machine parsing.

### Alerting Rules

```yaml
groups:
- name: sglang-gateway
  rules:
  - alert: HighErrorRate
    expr: |
      sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
      / sum(rate(smg_http_responses_total[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate on SGLang Gateway"

  - alert: CircuitBreakerOpen
    expr: count(smg_worker_cb_state == 1) > 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Worker circuit breaker is open"

  - alert: HighLatency
    expr: |
      histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le)) > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 latency exceeds 30 seconds"

  - alert: NoHealthyWorkers
    expr: sum(smg_worker_pool_size) == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "No healthy workers available"
```

---

## Troubleshooting

### Workers Never Ready

**Symptoms**: `/readiness` returns 503, `/workers` shows workers in `pending` or `processing` state indefinitely.

**Causes and Fixes**:

1. Workers not responding to health probes
   - Increase `--health-check-timeout-secs` (default 5)
   - Check worker health endpoint: `curl http://worker:8000/health`

2. Large models taking too long to load
   - Increase `--worker-startup-timeout-secs` (default 1800)
   - Monitor `/workers/{worker_id}` for job progress

3. Network connectivity issues
   - Verify DNS resolution: `nslookup worker-hostname`
   - Check firewall rules between gateway and workers

### Load Imbalance / Hot Workers

**Symptoms**: Uneven `smg_router_requests_total` across workers, high latency on some workers.

**Diagnosis**:

```bash
# Check load distribution
curl http://localhost:30000/get_loads

# Prometheus query
smg_router_requests_total by (worker_id)
```

**Fixes**:

1. Tune cache-aware thresholds:
   - Lower `--balance-abs-threshold` (default 64) for more aggressive rebalancing
   - Lower `--balance-rel-threshold` (default 1.5) to detect imbalance sooner
   - Lower `--cache-threshold` (default 0.3) to be less aggressive about cache affinity

2. Switch to a load-aware policy:
   ```bash
   --policy power_of_two  # Good for heterogeneous workloads
   ```

### Circuit Breaker Flapping

**Symptoms**: Workers rapidly cycling between healthy/unhealthy, intermittent 503 errors.

**Diagnosis**:
```promql
sum(rate(smg_worker_cb_transitions_total[5m])) by (worker_id, from_state, to_state)
```

**Fixes**:

1. Increase `--cb-failure-threshold` (default 10) to tolerate more failures
2. Extend `--cb-timeout-duration-secs` (default 60) for longer recovery windows
3. Extend `--cb-window-duration-secs` (default 120) for broader failure counting
4. Temporarily disable circuit breaker: `--disable-circuit-breaker`

### Queue Overflow (429 Responses)

**Symptoms**: Clients receiving 429 Too Many Requests.

**Fixes**:

1. Increase `--queue-size` (default 100)
2. Increase `--max-concurrent-requests` to allow more concurrent processing
3. Increase `--queue-timeout-secs` (default 60) for longer wait tolerance
4. Add more workers to increase throughput capacity

### Memory Growth

**Symptoms**: Steadily increasing memory usage over time.

**Causes**: Unbounded radix tree growth in cache-aware policy.

**Fixes**:

1. Reduce `--max-tree-size` (default 67108864)
2. Lower `--eviction-interval` (default 120) for more aggressive pruning
3. Monitor tree size via debug logging: `--log-level debug`

### gRPC Connection Issues

**Symptoms**: Connection refused, streaming errors.

**Fixes**:

1. Ensure workers are started with `--grpc-mode`
2. Verify `--model-path` or `--tokenizer-path` is provided to the router (required for gRPC mode)
3. Check that worker gRPC port matches the URL: `grpc://worker:20000`

### Tokenizer Loading Failures

**Symptoms**: Errors during startup or `/v1/tokenize` requests.

**Fixes**:

1. For private models, set `HF_TOKEN` environment variable for HuggingFace authentication
2. Verify local tokenizer paths are accessible from the gateway process
3. Check `/v1/tokenizers/{id}/status` for async loading errors

### Debug Mode

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs
```

Debug logging includes:
- Policy selection decisions
- Worker health check details
- Cache tree operations
- Retry attempts with backoff durations
- Request routing decisions

### Connection Mode Detection

The gateway automatically detects the connection mode based on worker URL prefixes:

```rust
fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
    for url in worker_urls {
        if url.starts_with("grpc://") || url.starts_with("grpcs://") {
            return ConnectionMode::Grpc { port: None };
        }
    }
    ConnectionMode::Http
}
```

All workers in a single deployment must use the same connection mode. Mixed HTTP/gRPC worker pools are not supported.
