# ONNX Runtime Reference - Chapter 46: Allocator and Memory Management

This chapter covers ONNX Runtime's memory management system in detail, including the IAllocator interface, arena allocators, GPU memory management, memory pattern optimization, and custom allocator integration.

---

## 46.1 IAllocator Interface

### 46.1.1 Base Allocator Interface

```cpp
// onnxruntime/core/framework/allocator.h
class IAllocator {
public:
    // Constructor with memory type and device
    IAllocator(OrtMemType mem_type, OrtDevice device)
        : mem_type_{mem_type}, device_{device} {}

    // Constructor with optional device ID
    IAllocator(OrtMemType mem_type, OrtDevice::DeviceType device_type,
               OrtDevice::MemoryType memory_type = OrtDevice::MemType::DEFAULT,
               int device_id = 0)
        : mem_type_{mem_type},
          device_{device_type, memory_type, device_id} {}

    virtual ~IAllocator() = default;

    // Core allocation/deallocation
    virtual void* Alloc(size_t size) = 0;
    virtual void Free(void* ptr) = 0;

    // Optional: allocation with byte alignment
    virtual void* AllocAligned(size_t size, size_t alignment) {
        // Default implementation ignores alignment
        return Alloc(size);
    }

    // Check if this allocator allows memory reservation (arena)
    virtual bool AllowsMemoryReservation() const { return false; }

    // Memory statistics
    virtual size_t Used() const { return 0; }
    virtual size_t Allocated() const { return 0; }
    virtual size_t Reserved() const { return 0; }

    // Attempt to shrink memory usage (return bytes freed)
    virtual size_t Shrink() { return 0; }

    // Get a human-readable name for debugging
    virtual const std::string& Info() const { return info_; }

    // Accessors
    OrtMemType MemType() const { return mem_type_; }
    const OrtDevice& GetDevice() const { return device_; }
    int DeviceId() const { return device_.Id(); }

    // Device type checks
    bool IsCPU() const {
        return device_.Type() == OrtDevice::CPU;
    }
    bool IsGPU() const {
        return device_.Type() == OrtDevice::GPU;
    }

    // Adapter info for GPU allocators
    virtual void* GetAdapter() const { return nullptr; }

protected:
    OrtMemType mem_type_;
    OrtDevice device_;
    std::string info_;
};

using AllocatorPtr = std::shared_ptr<IAllocator>;
```

### 46.1.2 OrtDevice Structure

```cpp
// onnxruntime/core/session/onnxruntime_c_api.h
struct OrtDevice {
    // Device types
    enum DeviceType : uint8_t {
        CPU = 0,
        GPU = 1,
        NPU = 2,   // Neural Processing Unit
        FPGA = 3,
    };

    // Memory types
    enum MemoryType : uint8_t {
        DEFAULT = 0,
        CUDA_PINNED = 1,   // Pinned (page-locked) host memory for CUDA
        HIP_PINNED = 2,    // Pinned host memory for ROCm/HIP
    };

    OrtDevice(DeviceType type, MemoryType mem_type, int id)
        : type_{type}, mem_type_{mem_type}, id_{static_cast<uint16_t>(id)} {}

    DeviceType Type() const { return type_; }
    MemoryType MemType() const { return mem_type_; }
    int Id() const { return id_; }

    bool operator==(const OrtDevice& other) const {
        return type_ == other.type_ &&
               mem_type_ == other.mem_type_ &&
               id_ == other.id_;
    }

    bool operator!=(const OrtDevice& other) const {
        return !(*this == other);
    }

    std::string ToString() const {
        return fmt::format("Device(type={}, mem_type={}, id={})",
                          type_, mem_type_, id_);
    }

private:
    DeviceType type_;
    MemoryType mem_type_;
    uint16_t id_;
};
```

### 46.1.3 OrtMemType Enumeration

```cpp
enum OrtMemType {
    OrtMemType_CPUInput = -2,        // CPU tensor, indexed from end
    OrtMemType_CPUOutput = -1,       // CPU output tensor
    OrtMemType_CPU = -1,             // Same as CPUOutput (alias)
    OrtMemType_Default = 0,          // Default device memory
    OrtMemType_User = 1,             // User-provided memory
};
```

---

## 46.2 CPUAllocator Implementation

### 46.2.1 CPUAllocator Class

```cpp
// onnxruntime/core/framework/cpu_allocator.h
class CPUAllocator : public IAllocator {
public:
    explicit CPUAllocator(int device_id = 0)
        : IAllocator(OrtMemType::OrtMemType_CPU,
                     OrtDevice(OrtDevice::CPU,
                               OrtDevice::MemType::DEFAULT,
                               device_id)) {
        info_ = "CPU";
    }

    void* Alloc(size_t size) override {
        if (size == 0) return nullptr;

        void* ptr;
        // Use aligned allocation for better SIMD performance
        #ifdef _WIN32
            ptr = _aligned_malloc(size, kAllocAlignment);
        #else
            ptr = std::aligned_alloc(kAllocAlignment, AlignUp(size, kAllocAlignment));
        #endif

        if (ptr == nullptr) {
            ORT_THROW("CPUAllocator: failed to allocate ",
                       size, " bytes");
        }
        return ptr;
    }

    void Free(void* ptr) override {
        if (ptr == nullptr) return;
        #ifdef _WIN32
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    void* AllocAligned(size_t size, size_t alignment) override {
        if (size == 0) return nullptr;

        void* ptr;
        #ifdef _WIN32
            ptr = _aligned_malloc(size, alignment);
        #else
            ptr = std::aligned_alloc(alignment, AlignUp(size, alignment));
        #endif

        if (ptr == nullptr) {
            ORT_THROW("CPUAllocator: failed to allocate aligned memory");
        }
        return ptr;
    }

    bool AllowsMemoryReservation() const override { return false; }

    static constexpr size_t kAllocAlignment = 64;  // Cache line alignment

private:
    static size_t AlignUp(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
};
```

### 46.2.2 CPUAllocator with Memory Tracking

```cpp
// Debug/trackable CPU allocator
class TrackinCPUAllocator : public CPUAllocator {
public:
    void* Alloc(size_t size) override {
        void* ptr = CPUAllocator::Alloc(size);
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = size;
        total_allocated_ += size;
        peak_allocated_ = std::max(peak_allocated_, total_allocated_);
        allocation_count_++;
        return ptr;
    }

    void Free(void* ptr) override {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_allocated_ -= it->second;
            allocations_.erase(it);
        }
        CPUAllocator::Free(ptr);
    }

    size_t Used() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_allocated_;
    }

    void PrintStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "CPU Allocator Stats:\n"
                  << "  Current allocated: " << total_allocated_ << " bytes\n"
                  << "  Peak allocated: " << peak_allocated_ << " bytes\n"
                  << "  Active allocations: " << allocations_.size() << "\n"
                  << "  Total allocations: " << allocation_count_ << "\n";
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;
    size_t total_allocated_ = 0;
    size_t peak_allocated_ = 0;
    size_t allocation_count_ = 0;
};
```

---

## 46.3 BFCArena (Best-Fit with Coalescing) Arena Allocator

### 46.3.1 Overview

BFCArena is ONNX Runtime's primary memory management allocator for GPU (and optionally CPU) memory. It implements a Best-Fit with Coalescing (BFC) algorithm similar to TensorFlow's allocator.

Key features:
- **Arena-based**: Pre-allocates large blocks from the raw allocator, then sub-allocates
- **Best-fit**: Finds the smallest free chunk that fits the request
- **Coalescing**: Merges adjacent free chunks to reduce fragmentation
- **Memory tracking**: Detailed statistics on allocation patterns
- **Configurable growth**: Multiple strategies for arena expansion

### 46.3.2 BFCArena Class

```cpp
// onnxruntime/core/framework/bfc_arena.h
class BFCArena : public IAllocator {
public:
    // Arena extension strategies
    enum class ArenaExtendStrategy {
        kNextPowerOfTwo = 1,    // Double the allocation size each time
        kSameAsRequested = 2,   // Allocate exactly what was requested
    };

    BFCArena(std::unique_ptr<IAllocator> raw_allocator,
             const OrtArenaCfg& arena_config);

    ~BFCArena() override;

    // IAllocator interface
    void* Alloc(size_t size) override;
    void Free(void* ptr) override;
    bool AllowsMemoryReservation() const override { return true; }

    // Statistics
    size_t Used() const override;
    size_t Allocated() const override;
    size_t Reserved() const override;
    size_t Shrink() override;

    // Detailed stats
    ArenaStats GetStats() const;
    void DumpMemoryStats(std::ostream& os) const;

private:
    // Internal structures
    struct Chunk;
    struct Bin;

    // Core allocation methods
    void* AllocateRawInternal(size_t num_bytes, bool dump_log_on_failure);
    void  DeallocateRawInternal(void* ptr);
    Chunk* FindChunkPtr(size_t num_bytes, uint64_t freed_before);
    void  InsertFreeChunkIntoBin(Chunk* chunk);
    void  RemoveFreeChunkFromBin(Chunk* chunk);
    void  MergeFreeChunks(Chunk* chunk1, Chunk* chunk2);

    // Arena management
    bool ExtendArena(size_t num_bytes);
    size_t RoundedBytes(size_t bytes);

    // Raw allocator (e.g., cudaMalloc)
    std::unique_ptr<IAllocator> raw_allocator_;

    // Configuration
    ArenaExtendStrategy arena_extend_strategy_;
    size_t initial_chunk_size_;
    size_t max_dead_bytes_per_chunk_;
    size_t memory_limit_;

    // Chunk management
    std::unordered_map<void*, Chunk*> chunk_map_;
    std::vector<Chunk*> chunks_;
    std::vector<Bin*> bins_;           // Free lists organized by size

    // Statistics
    mutable std::mutex mutex_;
    size_t total_allocated_ = 0;       // Bytes currently in use
    size_t total_reserved_ = 0;        // Total bytes from raw allocator
    size_t peak_allocated_ = 0;        // Peak bytes in use
    size_t num_allocs_ = 0;           // Total allocation count
    size_t num_frees_ = 0;            // Total free count
};
```

### 46.3.3 Chunk Data Structure

```cpp
struct BFCArena::Chunk {
    // Memory location
    void* ptr;                    // Pointer to the start of this chunk
    size_t size;                  // Full size of this chunk (including headers)
    size_t requested_size;        // Size requested by the user

    // Linkage
    Chunk* prev = nullptr;        // Previous chunk in memory order
    Chunk* next = nullptr;        // Next chunk in memory order

    // Allocation state
    bool in_use = false;          // Is this chunk currently allocated?
    uint64_t allocation_count = 0; // Monotonic counter for ordering
    int bin_index = -1;           // Which free bin this chunk belongs to (-1 = in use)

    // Bin free list linkage
    Chunk* prev_free = nullptr;
    Chunk* next_free = nullptr;

    bool IsFree() const { return !in_use; }

    bool CanMergeWith(const Chunk& other) const {
        // Can merge if adjacent and both free
        return !in_use && !other.in_use &&
               (static_cast<char*>(ptr) + size == other.ptr ||
                static_cast<char*>(other.ptr) + other.size == ptr);
    }

    std::string ToString() const {
        return fmt::format(
            "Chunk(ptr={}, size={}, requested={}, in_use={}, bin={})",
            ptr, size, requested_size, in_use, bin_index);
    }
};
```

### 46.3.4 Bin Data Structure

```cpp
struct BFCArena::Bin {
    // Each bin holds free chunks of a specific size range
    // Bin i holds chunks of size [2^i, 2^(i+1))
    size_t bin_size;              // Size class of this bin
    Chunk* free_chunks = nullptr; // Head of free chunk list

    bool IsEmpty() const { return free_chunks == nullptr; }

    void PushFreeChunk(Chunk* chunk) {
        chunk->prev_free = nullptr;
        chunk->next_free = free_chunks;
        if (free_chunks) {
            free_chunks->prev_free = chunk;
        }
        free_chunks = chunk;
    }

    Chunk* PopFreeChunk() {
        if (free_chunks == nullptr) return nullptr;
        Chunk* chunk = free_chunks;
        free_chunks = chunk->next_free;
        if (free_chunks) {
            free_chunks->prev_free = nullptr;
        }
        chunk->next_free = nullptr;
        chunk->prev_free = nullptr;
        return chunk;
    }

    void RemoveFreeChunk(Chunk* chunk) {
        if (chunk->prev_free) {
            chunk->prev_free->next_free = chunk->next_free;
        } else {
            free_chunks = chunk->next_free;
        }
        if (chunk->next_free) {
            chunk->next_free->prev_free = chunk->prev_free;
        }
        chunk->prev_free = nullptr;
        chunk->next_free = nullptr;
    }
};
```

### 46.3.5 Arena Extension Strategies

```cpp
// Strategy 1: kNextPowerOfTwo
// Each new arena allocation doubles the size of the previous one
// This provides amortized O(1) allocation cost
//
// Example sequence:
// Request 100MB → Allocate 256MB (first allocation)
// Request 200MB → Use from existing arena (fits)
// Request 300MB → Extend arena by 512MB (doubled from 256MB)
// Request 600MB → Extend arena by 1024MB (doubled from 512MB)

size_t BFCArena::NextPowerOfTwoSize(size_t requested_size) {
    size_t current_arena_size = total_reserved_;
    size_t extend_size = std::max(initial_chunk_size_, current_arena_size);

    // Double until we have enough
    while (extend_size < requested_size) {
        extend_size *= 2;
    }

    // Don't exceed memory limit
    extend_size = std::min(extend_size, memory_limit_ - total_reserved_);
    return extend_size;
}

// Strategy 2: kSameAsRequested
// Allocate exactly the requested size
// More memory-efficient but potentially more fragmentation
//
// Example sequence:
// Request 100MB → Allocate 100MB
// Request 200MB → Allocate 200MB
// Request 300MB → Allocate 300MB

size_t BFCArena::SameAsRequestedSize(size_t requested_size) {
    size_t extend_size = std::max(requested_size, initial_chunk_size_);
    extend_size = std::min(extend_size, memory_limit_ - total_reserved_);
    return extend_size;
}
```

### 46.3.6 Allocation Flow

```
User calls Alloc(size)
    │
    ├── 1. Round up size to alignment boundary
    │     RoundedBytes(size) → aligned_size
    │
    ├── 2. Search free bins for best-fit chunk
    │     FindChunkPtr(aligned_size) → chunk
    │     │
    │     ├── Found free chunk:
    │     │   ├── If chunk is much larger than needed (> max_dead_bytes):
    │     │   │   └── Split chunk into two: allocated + remainder
    │     │   └── Mark chunk as in_use
    │     │
    │     └── No suitable free chunk:
    │         ├── Try to extend arena
    │         │   ├── ExtendArena(aligned_size)
    │         │   │   ├── Determine extension size (strategy-dependent)
    │         │   │   ├── raw_allocator_->Alloc(extension_size)
    │         │   │   ├── Create new Chunk for extended memory
    │         │   │   └── Retry allocation
    │         │   │
    │         │   └── If extension fails:
    │         │       ├── Try smaller chunks
    │         │       └── Throw out-of-memory error
    │         │
    │         └── Try to coalesce adjacent free chunks
    │             └── MergeFreeChunks()
    │
    ├── 3. Update statistics
    │     total_allocated_ += chunk->size
    │     peak_allocated_ = max(peak, total_allocated)
    │     num_allocs_++
    │
    └── 4. Return chunk->ptr
```

### 46.3.7 Deallocation Flow

```
User calls Free(ptr)
    │
    ├── 1. Look up chunk by pointer
    │     chunk_map_[ptr] → chunk
    │
    ├── 2. Mark chunk as free
    │     chunk->in_use = false
    │     chunk->requested_size = 0
    │
    ├── 3. Try to merge with adjacent free chunks
    │     ├── Merge with previous chunk (if free)
    │     │   RemoveFreeChunkFromBin(prev_chunk)
    │     │   Merge: new chunk = prev + current
    │     │
    │     └── Merge with next chunk (if free)
    │         RemoveFreeChunkFromBin(next_chunk)
    │         Merge: new chunk = current + next
    │
    ├── 4. Insert merged chunk into appropriate bin
    │     bin_index = ComputeBinIndex(merged_chunk->size)
    │     InsertFreeChunkIntoBin(merged_chunk)
    │
    └── 5. Update statistics
          total_allocated_ -= chunk->size
          num_frees_++
```

### 46.3.8 BFCArena Implementation Details

```cpp
BFCArena::BFCArena(std::unique_ptr<IAllocator> raw_allocator,
                   const OrtArenaCfg& arena_config)
    : IAllocator(raw_allocator->MemType(), raw_allocator->GetDevice()),
      raw_allocator_(std::move(raw_allocator)) {

    // Parse configuration
    switch (arena_config.arena_extend_strategy) {
        case 1:
            arena_extend_strategy_ = ArenaExtendStrategy::kNextPowerOfTwo;
            break;
        case 2:
            arena_extend_strategy_ = ArenaExtendStrategy::kSameAsRequested;
            break;
        default:
            arena_extend_strategy_ = ArenaExtendStrategy::kNextPowerOfTwo;
    }

    memory_limit_ = arena_config.max_memory;
    initial_chunk_size_ = arena_config.initial_chunk_size_bytes > 0
                              ? arena_config.initial_chunk_size_bytes
                              : kDefaultInitialChunkSize;  // 1MB
    max_dead_bytes_per_chunk_ = arena_config.max_dead_bytes_per_chunk > 0
                                    ? arena_config.max_dead_bytes_per_chunk
                                    : kDefaultMaxDeadBytes;  // 128MB

    // Initialize bins (power-of-two size classes)
    // Typically 256 bins for sizes from 256 bytes to many GB
    constexpr int kNumBins = 256;
    bins_.reserve(kNumBins);
    for (int i = 0; i < kNumBins; ++i) {
        auto bin = new Bin();
        bin->bin_size = static_cast<size_t>(1) << i;
        bins_.push_back(bin);
    }
}

void* BFCArena::Alloc(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    return AllocateRawInternal(size, true);
}

void* BFCArena::AllocateRawInternal(size_t num_bytes, bool dump_log_on_failure) {
    // Round up to alignment
    num_bytes = RoundedBytes(num_bytes);

    if (num_bytes == 0) {
        num_bytes = kMinAllocationSize;  // 256 bytes minimum
    }

    // Try to find a free chunk
    Chunk* chunk = FindChunkPtr(num_bytes, 0);

    if (chunk == nullptr) {
        // No free chunk found, try to extend the arena
        if (ExtendArena(num_bytes)) {
            chunk = FindChunkPtr(num_bytes, 0);
        }
    }

    if (chunk == nullptr) {
        // Still no chunk, try to coalesce and retry
        // ... coalescing logic ...
        chunk = FindChunkPtr(num_bytes, 0);
    }

    if (chunk == nullptr) {
        if (dump_log_on_failure) {
            DumpMemoryStats(std::cerr);
        }
        ORT_THROW("BFCArena::AllocateRawInternal: out of memory. "
                   "Requested: ", num_bytes, " bytes. "
                   "Total allocated: ", total_allocated_, " bytes. "
                   "Memory limit: ", memory_limit_, " bytes.");
    }

    // Mark chunk as in use
    chunk->in_use = true;
    chunk->requested_size = num_bytes;
    chunk->allocation_count = ++allocation_counter_;

    // Update statistics
    total_allocated_ += chunk->size;
    peak_allocated_ = std::max(peak_allocated_, total_allocated_);
    num_allocs_++;

    return chunk->ptr;
}

void BFCArena::Free(void* ptr) {
    if (ptr == nullptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = chunk_map_.find(ptr);
    if (it == chunk_map_.end()) {
        ORT_THROW("BFCArena::Free: invalid pointer: ", ptr);
    }

    DeallocateRawInternal(ptr);
}

void BFCArena::DeallocateRawInternal(void* ptr) {
    auto it = chunk_map_.find(ptr);
    Chunk* chunk = it->second;

    // Update statistics
    total_allocated_ -= chunk->size;
    num_frees_++;

    // Mark as free
    chunk->in_use = false;
    chunk->requested_size = 0;

    // Try to merge with previous chunk
    if (chunk->prev && chunk->prev->IsFree()) {
        Chunk* prev = chunk->prev;
        RemoveFreeChunkFromBin(prev);
        MergeFreeChunks(prev, chunk);
        chunk = prev;
    }

    // Try to merge with next chunk
    if (chunk->next && chunk->next->IsFree()) {
        Chunk* next = chunk->next;
        RemoveFreeChunkFromBin(next);
        MergeFreeChunks(chunk, next);
    }

    // Insert back into free bin
    InsertFreeChunkIntoBin(chunk);
}

bool BFCArena::ExtendArena(size_t num_bytes) {
    // Determine extension size
    size_t extend_size;
    switch (arena_extend_strategy_) {
        case ArenaExtendStrategy::kNextPowerOfTwo:
            extend_size = NextPowerOfTwoSize(num_bytes);
            break;
        case ArenaExtendStrategy::kSameAsRequested:
            extend_size = SameAsRequestedSize(num_bytes);
            break;
    }

    // Check memory limit
    if (total_reserved_ + extend_size > memory_limit_) {
        // Try minimum possible
        extend_size = num_bytes;
        if (total_reserved_ + extend_size > memory_limit_) {
            return false;
        }
    }

    // Allocate from raw allocator
    void* new_mem = raw_allocator_->Alloc(extend_size);
    if (new_mem == nullptr) {
        return false;
    }

    // Create new chunk for the extended memory
    auto* new_chunk = new Chunk();
    new_chunk->ptr = new_mem;
    new_chunk->size = extend_size;
    new_chunk->in_use = false;
    new_chunk->allocation_count = 0;

    // Link to existing chunks
    if (!chunks_.empty()) {
        Chunk* last = chunks_.back();
        last->next = new_chunk;
        new_chunk->prev = last;
    }

    chunks_.push_back(new_chunk);
    chunk_map_[new_mem] = new_chunk;

    // Insert into free bin
    InsertFreeChunkIntoBin(new_chunk);

    // Update reserved count
    total_reserved_ += extend_size;

    return true;
}

Chunk* BFCArena::FindChunkPtr(size_t num_bytes, uint64_t freed_before) {
    // Find the smallest bin that could hold this allocation
    int bin_index = ComputeBinIndex(num_bytes);

    // Search bins from smallest to largest
    for (int i = bin_index; i < static_cast<int>(bins_.size()); ++i) {
        Bin* bin = bins_[i];
        if (bin->IsEmpty()) continue;

        // Search free chunks in this bin (best-fit)
        Chunk* best_chunk = nullptr;
        Chunk* chunk = bin->free_chunks;
        while (chunk != nullptr) {
            if (chunk->size >= num_bytes) {
                if (freed_before > 0 && chunk->allocation_count >= freed_before) {
                    chunk = chunk->next_free;
                    continue;
                }
                if (best_chunk == nullptr || chunk->size < best_chunk->size) {
                    best_chunk = chunk;
                }
            }
            chunk = chunk->next_free;
        }

        if (best_chunk != nullptr) {
            RemoveFreeChunkFromBin(best_chunk);

            // Split if much larger than needed
            size_t leftover = best_chunk->size - num_bytes;
            if (leftover > kMinAllocationSize &&
                leftover <= max_dead_bytes_per_chunk_) {
                // Split the chunk
                auto* remainder = new Chunk();
                remainder->ptr = static_cast<char*>(best_chunk->ptr) + num_bytes;
                remainder->size = leftover;
                remainder->in_use = false;
                remainder->prev = best_chunk;
                remainder->next = best_chunk->next;

                if (best_chunk->next) {
                    best_chunk->next->prev = remainder;
                }
                best_chunk->next = remainder;
                best_chunk->size = num_bytes;

                chunk_map_[remainder->ptr] = remainder;
                chunks_.push_back(remainder);

                InsertFreeChunkIntoBin(remainder);
            }

            return best_chunk;
        }
    }

    return nullptr;
}

size_t BFCArena::RoundedBytes(size_t bytes) {
    // Round up to kMinAllocationSize (256 bytes)
    size_t rounded = ((bytes + kMinAllocationSize - 1) / kMinAllocationSize)
                     * kMinAllocationSize;
    return rounded;
}

int BFCArena::ComputeBinIndex(size_t size) {
    // Find the smallest power of 2 >= size
    // This gives the bin index
    int index = 0;
    size_t s = size;
    while (s > 1) {
        s >>= 1;
        index++;
    }
    return std::max(0, index);
}
```

---

## 46.4 StreamAwareBFCArena for GPU

### 46.4.1 Overview

The `StreamAwareBFCArena` extends BFCArena with stream-aware allocation, enabling better memory reuse across CUDA streams.

```cpp
// onnxruntime/core/framework/stream_safe_arena.h
class StreamAwareBFCArena : public BFCArena {
public:
    StreamAwareBFCArena(std::unique_ptr<IAllocator> raw_allocator,
                        const OrtArenaCfg& arena_config);

    void* Alloc(size_t size) override;
    void Free(void* ptr) override;

    // Stream-aware allocation
    void* Alloc(size_t size, OrtStreamHandle stream);
    void Free(void* ptr, OrtStreamHandle stream);

    // Cross-stream synchronization
    void RecordStream(void* ptr, OrtStreamHandle stream);
    bool CanAllocate(void* ptr) const;

private:
    // Track which stream each chunk was last used on
    struct StreamChunkInfo {
        OrtStreamHandle last_stream;
        uint64_t last_use_count;
        std::vector<OrtStreamHandle> pending_streams;
    };

    std::unordered_map<void*, StreamChunkInfo> stream_info_;
    std::mutex stream_mutex_;
};
```

### 46.4.2 Stream-Aware Allocation Logic

```
Alloc(size, stream)
    │
    ├── 1. Find free chunk (same as BFCArena)
    │
    ├── 2. Check if chunk is safe to reuse
    │     ├── Was chunk last used on a different stream?
    │     │   ├── Yes: Check if previous stream has finished
    │     │   │   ├── cudaEventQuery() → check completion
    │     │   │   ├── If not done: skip this chunk, try another
    │     │   │   └── If done: safe to reuse
    │     │   └── No (same stream): safe to reuse
    │     └── Record new stream association
    │
    ├── 3. Record allocation event for synchronization
    │     cudaEventRecord(event, stream)
    │
    └── 4. Return chunk pointer
```

---

## 46.5 Arena Configuration (OrtArenaCfg)

### 46.5.1 OrtArenaCfg Structure

```cpp
// onnxruntime/core/session/onnxruntime_c_api.h
struct OrtArenaCfg {
    // Maximum memory that the arena can allocate (0 = unlimited)
    size_t max_memory;

    // Arena extension strategy
    // 1 = kNextPowerOfTwo (default)
    // 2 = kSameAsRequested
    int arena_extend_strategy;

    // Initial chunk size for the arena (0 = use default, 1MB)
    size_t initial_chunk_size_bytes;

    // Maximum dead bytes per chunk before splitting (0 = use default, 128MB)
    size_t max_dead_bytes_per_chunk;

    // Memory type for the arena
    OrtMemType memory_type;
};
```

### 46.5.2 Configuration Examples

```python
import onnxruntime as ort

# Default configuration
options = ort.SessionOptions()
session = ort.InferenceSession("model.onnx", options)

# Custom arena configuration
options = ort.SessionOptions()

# Limit GPU memory to 8GB
options.add_config_entry("session.gpu_memory_limit", "8589934592")  # 8GB in bytes

# Use kSameAsRequested strategy (more memory-efficient)
options.add_config_entry("session.arena_extend_strategy", "2")

# Set initial chunk size to 64MB
options.add_config_entry("session.arena_initial_chunk_size", "67108864")

# Set max dead bytes per chunk to 64MB
options.add_config_entry("session.arena_max_dead_bytes_per_chunk", "67108864")

# Disable arena (use direct allocation)
options.add_config_entry("session.use_arena", "0")

session = ort.InferenceSession("model.onnx", options,
    providers=["CUDAExecutionProvider"],
    provider_options=[{
        "device_id": 0,
        "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8GB
        "arena_extend_strategy": "kSameAsRequested",
    }])
```

### 46.5.3 Configuration via C API

```c
// Configure arena via C API
OrtArenaCfg arena_cfg;
arena_cfg.max_memory = 8ULL * 1024 * 1024 * 1024;  // 8GB
arena_cfg.arena_extend_strategy = 1;                 // kNextPowerOfTwo
arena_cfg.initial_chunk_size_bytes = 64 * 1024 * 1024;  // 64MB
arena_cfg.max_dead_bytes_per_chunk = 64 * 1024 * 1024;  // 64MB

// Apply to session options
OrtSessionOptionsSetArenaConfig(session_options, &arena_cfg);
```

---

## 46.6 Memory Patterns Optimization

### 46.6.1 Overview

Memory pattern optimization analyzes the memory usage of an inference run and creates a plan for reusing memory buffers across nodes.

```
Before memory pattern:
┌─────────────────────────────────────────────┐
│  Buffer A (100MB)  │  Buffer B (100MB)      │  → 200MB total
└─────────────────────────────────────────────┘

After memory pattern (A and B have non-overlapping lifetimes):
┌─────────────────────────────────────────────┐
│  Buffer A/B (100MB) shared                  │  → 100MB total
└─────────────────────────────────────────────┘
```

### 46.6.2 Memory Pattern Generation

```cpp
// onnxruntime/core/framework/mem_pattern.h
class MemoryPattern {
public:
    // Allocate a block within the pattern
    size_t Alloc(size_t size, size_t alignment = 256);

    // Get the offset of a previously allocated block
    size_t GetOffset(size_t alloc_id) const;

    // Total size of the pattern
    size_t TotalSize() const { return total_size_; }

    // Peak memory usage
    size_t PeakSize() const { return peak_size_; }

private:
    struct AllocInfo {
        size_t offset;
        size_t size;
        size_t alloc_id;
    };

    std::vector<AllocInfo> allocations_;
    size_t total_size_ = 0;
    size_t peak_size_ = 0;
    size_t next_alloc_id_ = 0;
};

// Memory pattern planner
class MemoryPatternPlanner {
public:
    // Record that a tensor is allocated
    void TraceAllocate(int mlvalue_idx, size_t size);

    // Record that a tensor is freed
    void TraceFree(int mlvalue_idx);

    // Generate the memory pattern
    std::unique_ptr<MemoryPattern> GeneratePattern();

private:
    struct TensorLifetime {
        int mlvalue_idx;
        size_t size;
        int alloc_node_index;     // Node index when allocated
        int free_node_index;      // Node index when freed
    };

    std::unordered_map<int, TensorLifetime> tensor_lifetimes_;
    std::vector<int> execution_order_;
};
```

### 46.6.3 Memory Pattern Application

```cpp
// onnxruntime/core/framework/execution_plan.h
class ExecutionPlan {
public:
    // Compute memory pattern for the entire graph
    Status ComputeMemoryPattern(const GraphViewer& graph,
                                 const std::vector<const Node*>& topo_sort);

    // Get the memory pattern
    const MemoryPattern& GetMemoryPattern() const { return pattern_; }

    // Allocate all buffers at once using the pattern
    Status AllocateMemoryPatternBuffer(AllocatorPtr allocator);

    // Get tensor buffer from pattern
    void* GetTensorBuffer(int mlvalue_idx) const;

private:
    MemoryPattern pattern_;
    void* pattern_buffer_ = nullptr;
    size_t pattern_buffer_size_ = 0;
    AllocatorPtr allocator_;
    std::unordered_map<int, size_t> mlvalue_to_offset_;
};
```

### 46.6.4 Enabling Memory Patterns

```python
import onnxruntime as ort

options = ort.SessionOptions()

# Enable memory pattern optimization (default: True)
options.enable_mem_pattern = True

# Disable for dynamic shapes (patterns are only effective with fixed shapes)
# options.enable_mem_pattern = False

# Enable memory reuse
options.add_config_entry("session.enable_mem_reuse", "1")

session = ort.InferenceSession("model.onnx", options)
```

---

## 46.7 Memory Reuse

### 46.7.1 Buffer Reuse Strategy

```cpp
// onnxruntime/core/framework/tensor.h
class Tensor {
public:
    // Tensors can share underlying buffers
    void ShareBufferWith(const Tensor& other) {
        buffer_ = other.buffer_;
        buffer_offset_ = other.buffer_offset_;
        // Shape and stride are independent
    }

    // Check if tensor owns its buffer
    bool OwnsBuffer() const {
        return buffer_.use_count() == 1;
    }

private:
    std::shared_ptr<TensorBuffer> buffer_;
    size_t buffer_offset_ = 0;
    TensorShape shape_;
    std::vector<size_t> strides_;
};

// Buffer allocator with reuse
class BufferAllocator {
public:
    // Try to reuse a previously freed buffer
    void* TryReuse(size_t size, size_t alignment) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Find a free buffer of matching size
        for (auto it = free_buffers_.begin(); it != free_buffers_.end(); ++it) {
            if (it->size >= size && it->size < size * 2) {
                void* ptr = it->ptr;
                free_buffers_.erase(it);
                return ptr;
            }
        }

        return nullptr;
    }

    void ReturnBuffer(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_buffers_.push_back({ptr, size});
    }

private:
    struct FreeBuffer {
        void* ptr;
        size_t size;
    };

    std::mutex mutex_;
    std::vector<FreeBuffer> free_buffers_;
};
```

---

## 46.8 Pre-packed Weights

### 46.8.1 Overview

Pre-packing transforms weights into hardware-optimal formats during session initialization, avoiding repeated packing during inference.

### 46.8.2 Weight Packing for Conv

```cpp
// onnxruntime/core/framework/conv_weights_packing.h
class ConvWeightsPacking {
public:
    // Pack Conv weights for optimal cache access
    static Tensor PackWeights(const Tensor& weights,
                               int64_t group,
                               bool is_1d,
                               bool is_depthwise) {
        auto shape = weights.Shape();
        int64_t output_channels = shape[0];
        int64_t input_channels_per_group = shape[1];

        if (is_depthwise) {
            // Depthwise Conv: pack for cache-friendly access
            return PackDepthwiseWeights(weights, group);
        } else {
            // Regular Conv: pack for vectorized access
            return PackRegularWeights(weights, group);
        }
    }

private:
    static Tensor PackRegularWeights(const Tensor& weights, int64_t group) {
        // Reorder weight dimensions for optimal SIMD access
        // Standard layout: [OC, IC, H, W]
        // Packed layout:   [OC/block, H, W, IC/block, block, block]
        // (block size depends on ISA: AVX2=6, AVX512=14)

        #ifdef USE_MLAS
        // Use MLAS packing routines
        auto packed = MLAS convolution weight packing
        #endif
        // ...
    }
};

// Pre-packing is done during session initialization
Status SessionState::PrePackWeights(const GraphViewer& graph) {
    for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "Conv") {
            // Get weight initializer
            auto weight_name = node.InputDefs()[1]->Name();
            auto weight_tensor = GetInitializer(weight_name);

            // Pack the weight
            auto packed = ConvWeightsPacking::PackWeights(
                *weight_tensor, group, is_1d, is_depthwise);

            // Store packed weight
            packed_weights_[weight_name] = std::move(packed);
        }

        if (node.OpType() == "MatMul") {
            // Pack MatMul B matrix for better cache usage
            auto b_name = node.InputDefs()[1]->Name();
            auto b_tensor = GetInitializer(b_name);

            // Transpose or block-pack for GEMM
            auto packed = MatMulPacking::Pack(*b_tensor);
            packed_weights_[b_name] = std::move(packed);
        }
    }

    return Status::OK();
}
```

### 46.8.3 Pre-packed Weights Configuration

```python
import onnxruntime as ort

options = ort.SessionOptions()

# Enable pre-packed weights (default: True)
options.add_config_entry("session.enable_prepacking", "1")

# Disable for models with dynamic weights
# options.add_config_entry("session.enable_prepacking", "0")
```

---

## 46.9 Memory Planning for Inference

### 46.9.1 Memory Planning Pipeline

```
1. Graph Analysis
   ├── Identify tensor lifetimes (alloc/free points)
   ├── Compute tensor sizes (with shape inference)
   └── Identify shared weight tensors

2. Memory Pattern Computation
   ├── Assign memory offsets to tensors
   ├── Overlapping buffers for non-overlapping lifetimes
   └── Minimize total memory footprint

3. Buffer Allocation
   ├── Allocate one large buffer for the pattern
   └── Map tensor offsets within the buffer

4. Weight Pre-packing
   ├── Transform weights for optimal access
   └── Store in separate or shared buffer
```

### 46.9.2 Memory Plan Data Structure

```cpp
// onnxruntime/core/framework/memory_planner.h
class MemoryPlanner {
public:
    struct MemoryPlan {
        // Per-device memory requirements
        struct DevicePlan {
            OrtDevice device;
            size_t total_size;
            std::unordered_map<int, size_t> tensor_offsets;
        };

        std::vector<DevicePlan> device_plans;

        // Total memory across all devices
        size_t TotalMemory() const {
            size_t total = 0;
            for (const auto& plan : device_plans) {
                total += plan.total_size;
            }
            return total;
        }
    };

    // Generate a memory plan from the graph
    static Status Plan(const GraphViewer& graph,
                        const std::vector<const Node*>& execution_order,
                        const std::unordered_map<int, TensorShape>& shapes,
                        MemoryPlan& plan);

private:
    // Interval scheduling for buffer reuse
    struct AllocationInterval {
        int mlvalue_idx;
        size_t size;
        size_t start_node;   // First node that uses this tensor
        size_t end_node;     // Last node that uses this tensor
    };

    // Assign offsets using first-fit decreasing
    static void AssignOffsets(
        std::vector<AllocationInterval>& intervals,
        size_t& total_size,
        std::unordered_map<int, size_t>& offsets);
};
```

---

## 46.10 Custom Allocator Registration

### 46.10.1 Registering Custom Allocators

```cpp
// C++ API for custom allocator
class CustomAllocator : public IAllocator {
public:
    CustomAllocator(int device_id)
        : IAllocator(OrtMemType_Default,
                     OrtDevice(OrtDevice::GPU,
                               OrtDevice::MemType::DEFAULT,
                               device_id)) {}

    void* Alloc(size_t size) override {
        // Custom allocation logic
        void* ptr = my_device_alloc(size);
        track_allocation(ptr, size);
        return ptr;
    }

    void Free(void* ptr) override {
        track_deallocation(ptr);
        my_device_free(ptr);
    }
};

// Register with session
OrtSessionOptionsAddCustomAllocator(options, allocator);
```

### 46.10.2 C API Registration

```c
// Define C allocator wrapper
static OrtAllocator* g_custom_allocator = nullptr;

OrtStatus* RegisterCustomAllocator(OrtEnv* env, OrtSessionOptions* options) {
    // Create custom allocator
    OrtAllocatorParams params = {
        .Alloc = [](void* ctx, size_t size) -> void* {
            return my_alloc(size);
        },
        .Free = [](void* ctx, void* ptr) {
            my_free(ptr);
        },
        .Info = [](void* ctx) -> const char* {
            return "CustomAllocator";
        },
        .ctx = nullptr,
    };

    return OrtSessionOptionsAddCustomAllocator(options, &params);
}
```

### 46.10.3 Python API Registration

```python
import onnxruntime as ort
import ctypes

# Load custom allocator from shared library
custom_alloc_lib = ctypes.CDLL("libcustom_allocator.so")

# Register via session options
options = ort.SessionOptions()
options.register_custom_allocator("CustomGPUAllocator", device_id=0)

session = ort.InferenceSession("model.onnx", options,
    providers=["CUDAExecutionProvider"])
```

---

## 46.11 Memory Statistics (GetStats)

### 46.11.1 ArenaStats Structure

```cpp
struct ArenaStats {
    // Allocation counts
    uint64_t num_allocs = 0;           // Total allocations
    uint64_t num_frees = 0;            // Total frees
    uint64_t num_resets = 0;           // Arena reset count

    // Memory usage
    int64_t total_allocated_bytes = 0; // Currently in use
    int64_t total_reserved_bytes = 0;  // Total from raw allocator
    int64_t peak_allocated_bytes = 0;  // Peak usage

    // Detailed tracking
    int64_t max_alloc_size = 0;        // Largest single allocation
    int64_t min_alloc_size = std::numeric_limits<int64_t>::max();
    double avg_alloc_size = 0.0;

    // Fragmentation metrics
    double fragmentation_ratio = 0.0;  // wasted / total
    int64_t free_chunk_count = 0;
    int64_t in_use_chunk_count = 0;

    // Timing
    uint64_t total_alloc_time_ns = 0;
    uint64_t total_free_time_ns = 0;
};
```

### 46.11.2 Getting Memory Statistics

```python
import onnxruntime as ort

options = ort.SessionOptions()
session = ort.InferenceSession("model.onnx", options,
    providers=["CUDAExecutionProvider"],
    provider_options=[{
        "device_id": 0,
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
    }])

# Run inference to collect stats
input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
session.run(None, input_data)

# Get memory profiling data
profile_data = session.end_profiling()
print(f"Profile saved to: {profile_data}")

# Get session memory info
memory_info = session.get_memory_info()
print(f"Peak memory: {memory_info['peak_allocated_bytes'] / 1024 / 1024:.1f} MB")
print(f"Current allocated: {memory_info['total_allocated_bytes'] / 1024 / 1024:.1f} MB")
print(f"Total reserved: {memory_info['total_reserved_bytes'] / 1024 / 1024:.1f} MB")
```

### 46.11.3 Dumping Memory State

```cpp
void BFCArena::DumpMemoryStats(std::ostream& os) const {
    os << "=== BFCArena Memory Stats ===" << std::endl;
    os << "Memory limit: " << memory_limit_ << " bytes" << std::endl;
    os << "Total reserved: " << total_reserved_ << " bytes" << std::endl;
    os << "Total allocated: " << total_allocated_ << " bytes" << std::endl;
    os << "Peak allocated: " << peak_allocated_ << " bytes" << std::endl;
    os << "Num allocs: " << num_allocs_ << std::endl;
    os << "Num frees: " << num_frees_ << std::endl;
    os << "Num chunks: " << chunks_.size() << std::endl;

    // Per-chunk details
    for (const auto* chunk : chunks_) {
        os << "  Chunk: ptr=" << chunk->ptr
           << " size=" << chunk->size
           << " in_use=" << chunk->in_use
           << " requested=" << chunk->requested_size
           << std::endl;
    }

    // Per-bin summary
    for (size_t i = 0; i < bins_.size(); ++i) {
        if (!bins_[i]->IsEmpty()) {
            int count = 0;
            Chunk* c = bins_[i]->free_chunks;
            while (c) { count++; c = c->next_free; }
            os << "  Bin[" << i << "] size_class="
               << bins_[i]->bin_size << " free_chunks=" << count << std::endl;
        }
    }
}
```

---

## 46.12 Shrink Operation

### 46.12.1 Overview

The shrink operation releases unused memory back to the system. This is useful for long-running sessions where peak memory usage may have been high but current usage is lower.

### 46.12.2 Shrink Implementation

```cpp
size_t BFCArena::Shrink() {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t bytes_freed = 0;

    // Find free chunks at the end of the arena that can be returned
    // to the raw allocator
    while (!chunks_.empty()) {
        Chunk* last = chunks_.back();

        if (last->IsFree()) {
            // Can return this chunk to the raw allocator
            size_t chunk_size = last->size;

            // Remove from free bin
            RemoveFreeChunkFromBin(last);

            // Unlink from chunk list
            if (last->prev) {
                last->prev->next = nullptr;
            }
            chunks_.pop_back();

            // Free from raw allocator
            raw_allocator_->Free(last->ptr);

            // Update stats
            total_reserved_ -= chunk_size;
            bytes_freed += chunk_size;

            delete last;
        } else {
            break;  // Can't shrink past in-use chunks
        }
    }

    return bytes_freed;
}
```

### 46.12.3 Triggering Shrink

```python
import onnxruntime as ort

# Trigger memory shrink after inference
session = ort.InferenceSession("model.onnx")

# Run a large inference
large_input = {"input": np.random.randn(100, 3, 224, 224).astype(np.float32)}
session.run(None, large_input)

# Shrink memory (release unused buffers)
session.shrink_memory()

# Now run a smaller inference
small_input = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
session.run(None, small_input)
```

---

## 46.13 Summary

| Component | Purpose | Key API |
|-----------|---------|---------|
| IAllocator | Base allocation interface | `Alloc()`, `Free()` |
| CPUAllocator | CPU memory with aligned allocation | `_aligned_malloc`, `std::aligned_alloc` |
| BFCArena | Arena allocator with BFC algorithm | Arena-based, bin-managed, coalescing |
| StreamAwareBFCArena | GPU stream-aware arena | Stream-safe reuse, cross-stream sync |
| OrtArenaCfg | Arena configuration | `max_memory`, `arena_extend_strategy`, etc. |
| MemoryPattern | Buffer reuse plan | Lifetime analysis, offset assignment |
| Pre-packed Weights | Optimized weight format | Conv weight packing, MatMul B packing |
| MemoryPlanner | Inference memory planning | Graph analysis, interval scheduling |
| GetStats | Memory statistics | Alloc counts, usage, fragmentation |
| Shrink | Memory release | Free trailing chunks |
