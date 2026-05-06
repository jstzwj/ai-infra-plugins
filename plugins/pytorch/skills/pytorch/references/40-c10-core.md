# c10 Core Library

## Overview

The `c10` library is PyTorch's foundational C++ library that provides core abstractions shared across all PyTorch components. The name "c10" stands for "Core 10" (or "Caffe2" backwards, as PyTorch merged with Caffe2). It contains essential types, utilities, and infrastructure that ATen, the autograd engine, and the JIT compiler all depend on. The library is deliberately kept minimal and dependency-light, containing only the most fundamental building blocks.

**Source location**: `c10/` in the PyTorch source tree.

**Key design principles**:
- Minimal dependencies (no reliance on ATen or higher-level PyTorch components)
- Header-heavy for performance-critical abstractions
- Provides intrusive pointer support for reference-counted objects
- Cross-platform compatibility (Linux, macOS, Windows)

---

## c10::Device

`c10::Device` represents a compute device on which tensors are allocated and operations are executed. It encapsulates a device type and an optional device index.

### Header

```cpp
#include <c10/core/Device.h>
```

### Class Definition

```cpp
namespace c10 {

class Device {
public:
  // Constructors
  Device(DeviceType type, int16_t index = -1);
  explicit Device(const std::string& device_string);

  // Accessors
  DeviceType type() const noexcept;
  int16_t index() const noexcept;

  // Type checks
  bool is_cpu() const noexcept;
  bool is_cuda() const noexcept;
  bool is_xpu() const noexcept;
  bool is_mps() const noexcept;
  bool is_ipu() const noexcept;
  bool is_xla() const noexcept;
  bool is_hpu() const noexcept;
  bool is_ve() const noexcept;
  bool is_lazy() const noexcept;
  bool is_vulkan() const noexcept;
  bool is_metal() const noexcept;
  bool is_meta() const noexcept;
  bool is_mtia() const noexcept;
  bool is_privateuse1() const noexcept;

  // Utility
  bool has_index() const noexcept;
  operator bool() const noexcept;  // true if type != COMPILE_TIME_MAX_DEVICE_TYPES
  bool operator==(const Device& other) const noexcept;
  bool operator!=(const Device& other) const noexcept;
  std::string str() const;

  // Static
  static DeviceType parse_type(const std::string& device_string);
};

} // namespace c10
```

### Usage Examples

```cpp
// Construction
c10::Device cpu_device(c10::kCPU);            // CPU device (no index)
c10::Device cuda_device(c10::kCUDA, 0);       // CUDA device 0
c10::Device cuda_device1("cuda:1");           // CUDA device 1 from string
c10::Device xpu_device(c10::kXPU, 0);         // XPU device 0
c10::Device mps_device(c10::kMPS);            // Apple MPS device

// Type checks
cpu_device.is_cpu();    // true
cuda_device.is_cuda();  // true
xpu_device.is_xpu();    // true
mps_device.is_mps();    // true

// Index access
cuda_device.index();      // 0
cuda_device.has_index();  // true
cpu_device.has_index();   // false (index is -1)

// String representation
cuda_device.str();  // "cuda:0"
cpu_device.str();   // "cpu"
```

### Device String Parsing

The string constructor accepts formats like:
- `"cpu"` - CPU device
- `"cuda"` - CUDA device (default index 0)
- `"cuda:0"` - CUDA device 0
- `"cuda:1"` - CUDA device 1
- `"xpu:0"` - XPU device 0
- `"mps:0"` - MPS device

---

## c10::DeviceType

`c10::DeviceType` is an enum class that enumerates all supported compute device types in PyTorch.

### Header

```cpp
#include <c10/core/DeviceType.h>
```

### Enum Values

```cpp
namespace c10 {

enum class DeviceType : int16_t {
  CPU,           // CPU (host memory)
  CUDA,          // NVIDIA GPU
  XPU,           // Intel GPU (oneAPI Level Zero)
  MPS,           // Apple Metal Performance Shaders
  IPU,           // Graphcore IPU
  XLA,           // Google TPU / XLA devices
  HPU,           // Intel Habana Gaudi
  VE,            // NEC Vector Engine
  Lazy,          // Lazy tensor (deferred execution)
  Vulkan,        // Vulkan GPU
  Metal,         // Apple Metal
  Meta,          // Meta tensors (no data, shape only)
  MTIA,          // Meta Training and Inference Accelerator
  PrivateUse1,   // Extension device slot 1
  // ... additional private use slots
  COMPILE_TIME_MAX_DEVICE_TYPES
};

// Convenience constants
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kMPS = DeviceType::MPS;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kXLA = DeviceType::XLA;
constexpr DeviceType kHPU = DeviceType::HPU;
constexpr DeviceType kVE = DeviceType::VE;
constexpr DeviceType kLazy = DeviceType::Lazy;
constexpr DeviceType kVulkan = DeviceType::Vulkan;
constexpr DeviceType kMetal = DeviceType::Metal;
constexpr DeviceType kMeta = DeviceType::Meta;

} // namespace c10
```

### Utility Functions

```cpp
// Check if device type supports backend dispatch
bool isSparseDispatch(DeviceType type);

// Get string representation
const std::string& DeviceTypeName(DeviceType type, bool lower_case = false);

// Check device type validity
bool isValidDeviceType(DeviceType type);

// Device type to short string
std::string device_type_to_string(DeviceType type);
```

---

## c10::ScalarType

`c10::ScalarType` enumerates all supported data types for tensor elements.

### Header

```cpp
#include <c10/core/ScalarType.h>
```

### Enum Values

```cpp
namespace c10 {

enum class ScalarType : int8_t {
  // Floating point types
  Float64,      // double, 64-bit float
  Float32,      // float, 32-bit float
  Float16,      // half, 16-bit IEEE float
  BFloat16,     // bfloat16, 16-bit brain float
  Float8_e5m2,  // 8-bit float (E5M2 format)
  Float8_e4m3fn, // 8-bit float (E4M3FN format)
  Float8_e5m2fnuz, // 8-bit float (E5M2FNUZ format)
  Float8_e4m3fnuz, // 8-bit float (E4M3FNUZ format)
  Float8_e8m0fnu, // 8-bit float (E8M0 scale format)

  // Integer types
  Int64,        // int64_t, 64-bit signed integer
  Int32,        // int32_t, 32-bit signed integer
  Int16,        // int16_t, 16-bit signed integer
  Int8,         // int8_t, 8-bit signed integer
  UInt64,       // uint64_t, 64-bit unsigned integer
  UInt32,       // uint32_t, 32-bit unsigned integer
  UInt16,       // uint16_t, 16-bit unsigned integer
  UInt8,        // uint8_t, 8-bit unsigned integer

  // Boolean
  Bool,

  // Complex types
  ComplexHalf,   // complex<half>
  ComplexFloat,  // complex<float> (complex64)
  ComplexDouble, // complex<double> (complex128)

  // Quantized types
  QInt8,        // quantized 8-bit signed integer
  QUInt8,       // quantized 8-bit unsigned integer
  QInt32,       // quantized 32-bit signed integer
  QUInt4x2,     // quantized 4-bit unsigned (packed in pairs)
  QUInt2x4,     // quantized 2-bit unsigned (packed in quads)

  // Undefined
  Undefined,

  // Number of types
  NumOptions
};

// Convenience aliases
constexpr ScalarType kFloat64 = ScalarType::Float64;
constexpr ScalarType kFloat32 = ScalarType::Float32;
constexpr ScalarType kFloat16 = ScalarType::Float16;
constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
constexpr ScalarType kInt64 = ScalarType::Int64;
constexpr ScalarType kInt32 = ScalarType::Int32;
constexpr ScalarType kInt16 = ScalarType::Int16;
constexpr ScalarType kInt8 = ScalarType::Int8;
constexpr ScalarType kUInt8 = ScalarType::UInt8;
constexpr ScalarType kBool = ScalarType::Bool;
constexpr ScalarType kComplexHalf = ScalarType::ComplexHalf;
constexpr ScalarType kComplexFloat = ScalarType::ComplexFloat;
constexpr ScalarType kComplexDouble = ScalarType::ComplexDouble;
constexpr ScalarType kQInt8 = ScalarType::QInt8;
constexpr ScalarType kQUInt8 = ScalarType::QUInt8;
constexpr ScalarType kQInt32 = ScalarType::QInt32;

} // namespace c10
```

### Type Properties

```cpp
// Get element size in bytes
constexpr size_t elementSize(ScalarType type);

// Type checking functions
bool isFloatingType(ScalarType type);
bool isIntegerType(ScalarType type);
bool isSignedType(ScalarType type);
bool isUnsignedType(ScalarType type);
bool isComplexType(ScalarType type);
bool isQIntType(ScalarType type);
bool isSubByteType(ScalarType type);
bool isFloat8Type(ScalarType type);

// Type promotion
ScalarType promoteTypes(ScalarType a, ScalarType b);

// C++ type to ScalarType mapping
template<typename T>
struct scalar_type_to_c_type;

// ScalarType to C++ type mapping
template<ScalarType>
struct scalar_type_to_c_type;
```

### Type Correspondence Table

| ScalarType         | C++ Type          | Python Type       | Size (bytes) |
|---------------------|-------------------|-------------------|--------------|
| kFloat64           | double            | torch.float64     | 8            |
| kFloat32           | float             | torch.float32     | 4            |
| kFloat16           | c10::Half         | torch.float16     | 2            |
| kBFloat16          | c10::BFloat16     | torch.bfloat16    | 2            |
| kInt64             | int64_t           | torch.int64       | 8            |
| kInt32             | int32_t           | torch.int32       | 4            |
| kInt16             | int16_t           | torch.int16       | 2            |
| kInt8              | int8_t            | torch.int8        | 1            |
| kUInt8             | uint8_t           | torch.uint8       | 1            |
| kBool              | bool              | torch.bool        | 1            |
| kComplexFloat      | c10::complex<float> | torch.complex64  | 8            |
| kComplexDouble     | c10::complex<double>| torch.complex128 | 16           |

---

## c10::Scalar

`c10::Scalar` represents a single scalar value that can be one of several types: boolean, integer, floating-point, or complex. It is used throughout PyTorch as a parameter type for operations that accept scalar arguments.

### Header

```cpp
#include <c10/core/Scalar.h>
```

### Class Definition

```cpp
namespace c10 {

class Scalar {
public:
  // Constructors
  Scalar() : tag(Tag::HAS_NONE) {}
  Scalar(bool v);
  Scalar(int64_t v);
  Scalar(double v);
  Scalar(c10::complex<double> v);
  Scalar(c10::Half v);
  Scalar(c10::BFloat16 v);

  // Template constructors
  template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
  Scalar(T v);

  // Type queries
  bool isBoolean() const noexcept;
  bool isIntegral(bool include_bool = false) const noexcept;
  bool isFloatingPoint() const noexcept;
  bool isComplex() const noexcept;
  bool isNull() const noexcept;

  // Value access (with type checking)
  bool toBoolean() const;
  int64_t toInt() const;
  double toDouble() const;
  c10::complex<double> toComplexDouble() const;

  // Value access (with conversion)
  template<typename T>
  T to() const;

  // Type of the scalar
  ScalarType type() const noexcept;

  // Assignment
  Scalar& operator=(const Scalar& other) = default;
};

} // namespace c10
```

### Usage Examples

```cpp
// Construction from various types
c10::Scalar s_bool(true);
c10::Scalar s_int(42);
c10::Scalar s_float(3.14);
c10::Scalar s_complex(c10::complex<double>(1.0, 2.0));

// Type queries
s_bool.isBoolean();      // true
s_int.isIntegral();       // true
s_float.isFloatingPoint(); // true
s_complex.isComplex();    // true

// Value access
s_bool.toBoolean();           // true
s_int.toInt();                // 42
s_float.toDouble();           // 3.14
s_complex.toComplexDouble();  // (1.0, 2.0)

// Generic access with conversion
s_int.to<float>();    // 42.0 (int -> float conversion)
s_float.to<int>();    // 3 (float -> int truncation)

// Used in tensor operations
torch::Tensor t = torch::ones({3, 3});
torch::Tensor result = t.add(c10::Scalar(5));  // add scalar 5
```

---

## c10::TensorImpl

`c10::TensorImpl` is the core implementation class underlying every PyTorch tensor. It holds the storage, shape, stride, and metadata for a tensor. This class uses `c10::intrusive_ptr` for reference counting.

### Header

```cpp
#include <c10/core/TensorImpl.h>
```

### Key Members

```cpp
namespace c10 {

class TensorImpl : public c10::intrusive_ptr_target {
public:
  // Storage access
  Storage storage() const;
  const Storage& unsafe_storage() const;
  void set_storage(Storage storage);

  // Shape and layout
  IntArrayRef sizes() const;
  IntArrayRef strides() const;
  int64_t dim() const;
  int64_t numel() const;
  int64_t storage_offset() const;

  // Data type and device
  ScalarType dtype() const;
  Device device() const;
  Layout layout() const;

  // Contiguity checks
  bool is_contiguous() const;
  bool is_contiguous(at::MemoryFormat memory_format) const;
  bool is_strides_like(at::MemoryFormat memory_format) const;
  bool is_non_overlapping_and_dense() const;

  // Device checks
  bool is_cpu() const;
  bool is_cuda() const;
  bool is_xpu() const;
  bool is_mps() const;

  // Data access
  void* data_ptr() const;
  template<typename T>
  T* data_ptr() const;
  void* raw_data() const;

  // Metadata
  bool requires_grad() const;
  void set_requires_grad(bool requires_grad);
  bool is_leaf() const;

  // Modification
  void set_sizes_contiguous(IntArrayRef sizes);
  void set_sizes_and_strides(IntArrayRef sizes, IntArrayRef strides);
  void set_storage_offset(int64_t offset);

  // Reshape (in-place)
  void resize(IntArrayRef sizes, IntArrayRef strides);
  void resize_(IntArrayRef sizes);

  // Type dispatch
  DispatchKeySet key_set() const;
  bool is_wrapped_number() const;

protected:
  c10::Storage storage_;
  c10::SmallVector<int64_t, 5> sizes_;
  c10::SmallVector<int64_t, 5> strides_;
  int64_t storage_offset_ = 0;
  ScalarType dtype_ = ScalarType::Undefined;
  // ... additional members
};

} // namespace c10
```

### Usage Notes

Direct interaction with `TensorImpl` is uncommon in user code. Most users interact through `at::Tensor`, which wraps `c10::intrusive_ptr<TensorImpl>`. However, understanding `TensorImpl` is essential for:
- Writing custom C++ kernels and operators
- Implementing new tensor backends
- Debugging tensor layout issues
- Optimizing memory access patterns

```cpp
// Access TensorImpl from a Tensor
at::Tensor tensor = torch::randn({3, 4});
c10::TensorImpl* impl = tensor.unsafeGetTensorImpl();

// Query properties
impl->sizes();           // {3, 4}
impl->strides();         // {4, 1}
impl->dim();             // 2
impl->numel();           // 12
impl->storage_offset();  // 0
impl->is_contiguous();   // true
impl->dtype();           // kFloat32
impl->device();          // cpu
```

---

## c10::Storage

`c10::Storage` manages the underlying data buffer for tensors. It wraps a `StorageImpl` via an `intrusive_ptr` and provides methods to access and manipulate raw data.

### Header

```cpp
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
```

### Class Definition

```cpp
namespace c10 {

class Storage {
public:
  // Constructors
  Storage() = default;
  Storage(StorageImpl* ptr);
  Storage(size_t size, ScalarType dtype, Allocator* allocator = nullptr);
  Storage(size_t size, at::DataPtr data, ScalarType dtype);

  // Capacity
  size_t nbytes() const;
  size_t itemsize() const;
  size_t numel() const;
  bool empty() const;

  // Data access
  void* data_ptr() const;
  void* mutable_data_ptr() const;
  at::DataPtr& data_ptr();

  // Modification
  void set_data_ptr(at::DataPtr&& data_ptr);
  void set_nbytes(size_t new_nbytes);
  void resize_(size_t new_numel);

  // Device
  Device device() const;

  // Type
  ScalarType dtype() const;

  // Underlying impl
  StorageImpl* unsafeGetStorageImpl() const;
};

} // namespace c10
```

### Usage Examples

```cpp
// Create storage
auto storage = c10::Storage(100, c10::kFloat32);

// Inspect storage
storage.numel();    // 100
storage.nbytes();   // 400 (100 * sizeof(float))
storage.itemsize(); // 4

// Access raw data
float* data = static_cast<float*>(storage.data_ptr());

// Resize storage
storage.resize_(200);
storage.numel();  // 200
```

---

## c10::TensorOptions

`c10::TensorOptions` bundles the common options used when creating tensors: dtype, device, layout, whether the tensor requires gradients, and whether memory is pinned.

### Header

```cpp
#include <c10/core/TensorOptions.h>
```

### Class Definition

```cpp
namespace c10 {

class TensorOptions {
public:
  // Constructors
  TensorOptions() = default;
  TensorOptions(ScalarType dtype, Layout layout, Device device,
                bool requires_grad = false, bool pinned_memory = false);

  // Builder-style setters
  TensorOptions dtype(ScalarType dtype) const;
  TensorOptions device(Device device) const;
  TensorOptions layout(Layout layout) const;
  TensorOptions requires_grad(bool requires_grad) const;
  TensorOptions pinned_memory(bool pinned_memory) const;

  // Accessors
  ScalarType dtype() const noexcept;
  Device device() const noexcept;
  Layout layout() const noexcept;
  bool requires_grad() const noexcept;
  bool pinned_memory() const noexcept;

  // Merge options (later options override)
  TensorOptions merge(TensorOptions other) const;

  // Convenience
  bool has_dtype() const noexcept;
  bool has_device() const noexcept;
  bool has_layout() const noexcept;
};

// Convenience constructors
TensorOptions dtype(ScalarType dtype);
TensorOptions device(Device device);
TensorOptions layout(Layout layout);

} // namespace c10
```

### Usage Examples

```cpp
// Create TensorOptions with all fields
auto options = c10::TensorOptions(c10::kFloat32, c10::kStrided,
                                  c10::Device(c10::kCUDA, 0));

// Builder-style construction
auto options2 = c10::TensorOptions()
    .dtype(c10::kFloat32)
    .device(c10::kCUDA)
    .layout(c10::kStrided)
    .requires_grad(true)
    .pinned_memory(false);

// Used in tensor creation
auto tensor = torch::empty({3, 4}, options);

// Merge with partial options
auto merged = options.dtype(c10::kFloat16);
// merged inherits device/layout from options, overrides dtype
```

---

## c10::ArrayRef<T>

`c10::ArrayRef<T>` is a non-owning reference to a contiguous array of elements. It is similar to C++20's `std::span` and is used extensively throughout PyTorch for passing array-like arguments without ownership transfer.

### Header

```cpp
#include <c10/util/ArrayRef.h>
```

### Class Definition

```cpp
namespace c10 {

template<typename T>
class ArrayRef {
public:
  // Constructors
  ArrayRef() noexcept;
  ArrayRef(const T* data, size_t length);
  ArrayRef(const T* begin, const T* end);
  ArrayRef(const T& single_element);
  ArrayRef(std::initializer_list<T> list);
  ArrayRef(const std::vector<T>& vec);
  ArrayRef(const SmallVectorImpl<T>& vec);

  // Iterators
  const T* begin() const noexcept;
  const T* end() const noexcept;

  // Element access
  const T* data() const noexcept;
  const T& operator[](size_t index) const;
  const T& front() const;
  const T& back() const;

  // Capacity
  size_t size() const noexcept;
  bool empty() const noexcept;

  // Sub-operations
  ArrayRef<T> slice(size_t start, size_t length) const;
  ArrayRef<T> drop_front(size_t n = 1) const;
  ArrayRef<T> drop_back(size_t n = 1) const;

  // Comparison
  bool equals(ArrayRef<T> other) const;

  // Conversion
  std::vector<T> vec() const;
};

// Type aliases used throughout PyTorch
using IntArrayRef = ArrayRef<int64_t>;
using DoubleArrayRef = ArrayRef<double>;

} // namespace c10
```

### Usage Examples

```cpp
// From various sources
std::vector<int64_t> vec = {1, 2, 3, 4};
c10::IntArrayRef ref1(vec);                // from vector
c10::IntArrayRef ref2({1, 2, 3, 4});       // from initializer list
c10::IntArrayRef ref3(vec.data(), 4);      // from pointer + length

// Passing to functions
void process_sizes(c10::IntArrayRef sizes);

// Used everywhere in tensor APIs
torch::Tensor t = torch::randn({3, 4, 5});
// The {3, 4, 5} is implicitly converted to IntArrayRef

// Slicing
c10::IntArrayRef full = {1, 2, 3, 4, 5};
auto sub = full.slice(1, 3);  // {2, 3, 4}
auto tail = full.drop_front(2); // {3, 4, 5}
```

---

## c10::SmallVector<T, N>

`c10::SmallVector<T, N>` provides small buffer optimization (SBO). For N or fewer elements, data is stored inline without heap allocation. This is critical for performance since most tensors have 5 or fewer dimensions.

### Header

```cpp
#include <c10/util/SmallVector.h>
```

### Definition

```cpp
namespace c10 {

template<typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
  // N elements stored inline without heap allocation
  // Automatically falls back to heap allocation when size > N
};

// Common specializations
using SmallVector<int64_t, 5> DimVector;    // tensor dimensions
using SmallVector<int64_t, 5> StrideVector; // tensor strides

} // namespace c10
```

### Key Features

```cpp
// Create with inline storage for up to 4 elements
c10::SmallVector<int64_t, 4> vec;

// Add elements (no heap allocation for first 4)
vec.push_back(1);  // inline
vec.push_back(2);  // inline
vec.push_back(3);  // inline
vec.push_back(4);  // inline
vec.push_back(5);  // heap allocation triggered

// Standard vector interface
vec.size();
vec.empty();
vec[0];
vec.begin();
vec.end();
vec.clear();
vec.resize(10);
```

### Why SmallVector

Most PyTorch tensors have 1-5 dimensions. Using `SmallVector<T, 5>` for sizes and strides avoids heap allocation in the common case, significantly improving performance for tensor creation and reshaping operations.

---

## c10::OptionalRef<T>

`c10::OptionalRef<T>` is a non-owning optional reference to an object. It is similar to `std::optional<std::reference_wrapper<T>>` but more ergonomic and efficient.

### Header

```cpp
#include <c10/util/OptionalRef.h>
```

### Usage

```cpp
// Create from reference
int value = 42;
c10::OptionalRef<int> ref(value);
c10::OptionalRef<int> empty;  // no reference

// Check and access
if (ref) {
  int& val = *ref;
  val = 100;  // modifies original
}

// has_value() check
ref.has_value();  // true
empty.has_value(); // false
```

---

## c10::Dict<Key, Value>

`c10::Dict` is an intrusive reference-counted dictionary type designed for use in the JIT/ TorchScript system. It supports iteration, lookup, insertion, and removal.

### Header

```cpp
#include <c10/core/Dict.h>
```

### Usage

```cpp
// Create a dict
c10::Dict<std::string, int64_t> dict;
dict.insert("key1", 1);
dict.insert("key2", 2);

// Lookup
auto val = dict.find("key1");  // iterator
bool has = dict.contains("key1");

// Iteration
for (auto& pair : dict) {
  auto key = pair.key();
  auto value = pair.value();
}

// Size
dict.size();
dict.empty();
```

---

## c10::List<T>

`c10::List` is an intrusive reference-counted list type, also primarily used in TorchScript. It wraps a `std::vector` with reference counting.

### Header

```cpp
#include <c10/core/List.h>
```

### Usage

```cpp
// Create a list
c10::List<int64_t> list;
list.push_back(1);
list.push_back(2);
list.push_back(3);

// Access
list[0];       // 1
list.size();   // 3
list.empty();  // false

// Modification
list.append(4);
list.insert(0, 0);

// Iteration
for (const auto& elem : list) {
  // ...
}
```

---

## c10::IValue

`c10::IValue` (Intrusive Value) is a tagged union type that can hold any of the types supported by the PyTorch JIT system. It is the fundamental value type used for passing arguments through the dispatcher and JIT.

### Header

```cpp
#include <c10/core/ivalue.h>
```

### Supported Types

```cpp
namespace c10 {

class IValue {
public:
  // Tag enumeration for contained type
  enum class Tag {
    None,
    Tensor,
    Storage,
    Double,
    Int,
    ComplexDouble,
    Bool,
    Tuple,
    String,
    Blob,
    GenericDict,
    GenericList,
    Future,
    RRef,
    Device,
    Object,
    Generator,
    Quantizer,
    PyObject,
    Enum,
    CustomClass,
    ListInt,
    ListDouble,
    ListBool,
    ListTensor,
    ListOptionalTensor,
    ListString,
    ListGenericList,
    ListGenericDict,
    // ...
  };

  // Constructors from various types
  IValue() : tag(Tag::None) {}
  IValue(c10::Tensor tensor);
  IValue(int64_t value);
  IValue(double value);
  IValue(bool value);
  IValue(c10::complex<double> value);
  IValue(const std::string& value);
  IValue(const char* value);
  IValue(c10::Device device);
  IValue(c10::Dict<IValue, IValue> dict);
  IValue(c10::List<IValue> list);
  IValue(std::tuple<...> tuple);

  // Type checking
  bool isNone() const;
  bool isTensor() const;
  bool isDouble() const;
  bool isInt() const;
  bool isBool() const;
  bool isComplexDouble() const;
  bool isString() const;
  bool isDevice() const;
  bool isList() const;
  bool isTuple() const;
  bool isDict() const;
  bool isGenericDict() const;
  bool isObject() const;
  bool isFuture() const;
  bool isEnum() const;
  bool isCustomClass() const;

  // Value extraction
  c10::Tensor toTensor() const;
  int64_t toInt() const;
  double toDouble() const;
  bool toBool() const;
  c10::complex<double> toComplexDouble() const;
  std::string toString() const;
  c10::Device toDevice() const;
  c10::List<IValue> toList() const;
  c10::Dict<IValue, IValue> toGenericDict() const;

  // Type name
  std::string type_name() const;

  // Tag access
  Tag tag() const noexcept;
};

} // namespace c10
```

### Usage Examples

```cpp
// Constructing IValues
c10::IValue iv_none;
c10::IValue iv_tensor(torch::randn({3, 4}));
c10::IValue iv_int(42);
c10::IValue iv_float(3.14);
c10::IValue iv_bool(true);
c10::IValue iv_str("hello");
c10::IValue iv_device(c10::Device(c10::kCUDA, 0));
c10::IValue iv_complex(c10::complex<double>(1.0, 2.0));

// Type checking and extraction
if (iv_tensor.isTensor()) {
  auto t = iv_tensor.toTensor();
}

if (iv_int.isInt()) {
  int64_t val = iv_int.toInt();
}

// Using in operator arguments
std::vector<c10::IValue> args = {
  c10::IValue(tensor),     // self
  c10::IValue(42),         // dim
  c10::IValue(true)        // keepdim
};
```

---

## c10::DispatchKey

`c10::DispatchKey` is an enum that identifies different dispatch components in PyTorch's operator dispatch system. Each key represents a layer of functionality that may handle an operator differently.

### Header

```cpp
#include <c10/core/DispatchKey.h>
```

### Key Categories

```cpp
namespace c10 {

enum class DispatchKey : uint16_t {
  // =============================================
  // Backend dispatch keys (where computation runs)
  // =============================================
  CPU,                    // CPU kernel
  CUDA,                   // CUDA kernel
  XPU,                    // Intel XPU kernel
  MPS,                    // Apple MPS kernel
  IPU,                    // Graphcore IPU
  XLA,                    // Google XLA
  HPU,                    // Intel Habana
  VE,                     // NEC Vector Engine
  Lazy,                   // Lazy tensor
  Vulkan,                 // Vulkan GPU
  Metal,                  // Apple Metal
  Meta,                   // Meta tensors (shape only)
  MTIA,                   // Meta Training/Inference Accelerator
  PrivateUse1,            // Extension backend 1
  PrivateUse2,            // Extension backend 2
  PrivateUse3,            // Extension backend 3

  // =============================================
  // Autograd dispatch keys
  // =============================================
  Autograd,               // Autograd for all backends
  AutogradCPU,            // Autograd specifically for CPU
  AutogradCUDA,           // Autograd specifically for CUDA
  AutogradXPU,            // Autograd specifically for XPU
  AutogradMPS,            // Autograd specifically for MPS
  AutogradIPU,            // Autograd specifically for IPU
  AutogradXLA,            // Autograd specifically for XLA
  AutogradHPU,            // Autograd specifically for HPU
  AutogradLazy,           // Autograd specifically for Lazy
  AutogradVE,             // Autograd specifically for VE
  AutogradVulkan,         // Autograd specifically for Vulkan
  AutogradMeta,           // Autograd specifically for Meta
  AutogradMTIA,           // Autograd specifically for MTIA
  AutogradPrivateUse1,    // Autograd specifically for PrivateUse1
  AutogradPrivateUse2,    // Autograd specifically for PrivateUse2
  AutogradPrivateUse3,    // Autograd specifically for PrivateUse3

  // =============================================
  // Functionality dispatch keys
  // =============================================
  Tracer,                 // JIT tracer
  AutocastCPU,            // Automatic mixed precision (CPU)
  AutocastCUDA,           // Automatic mixed precision (CUDA)
  FuncTorchDynamicLayer,  // functorch dynamic layer
  FuncTorchGradWrapper,   // functorch grad wrapper
  Functionalize,          // Functionalization pass
  ADInplaceOrView,        // Autograd inplace/view handling
  Python,                 // Python implementation
  PythonTLSSnapshot,      // Python TLS snapshot
  Dense,                  // Dense tensor (non-sparse)
  Sparse,                 // Sparse tensor
  SparseCsr,              // Sparse CSR tensor

  // =============================================
  // Composite dispatch keys
  // =============================================
  CompositeExplicitAutograd,        // Composite impl for explicit autograd
  CompositeExplicitAutogradNonFunctional,  // Non-functional composite
  CompositeImplicitAutograd,        // Composite impl for implicit autograd

  // =============================================
  // Special keys
  // =============================================
  Conjugate,              // Conjugate handling
  Negative,               // Negative handling
  ZeroTensor,             // Zero tensor optimization
  Complex,                // Complex number handling
  BackendSelect,          // Backend selection logic
  Named,                  // Named tensor handling

  // =============================================
  // Alias analysis keys
  // =============================================
  AliasAnalysis,          // Alias analysis information
};

} // namespace c10
```

### Dispatch Key Set

```cpp
#include <c10/core/DispatchKeySet.h>

// DispatchKeySet represents a set of dispatch keys
class DispatchKeySet {
public:
  DispatchKeySet() = default;
  explicit DispatchKeySet(DispatchKey key);
  DispatchKeySet(std::initializer_list<DispatchKey> keys);

  // Set operations
  bool has(DispatchKey key) const;
  DispatchKeySet add(DispatchKey key) const;
  DispatchKeySet remove(DispatchKey key) const;

  // Highest priority key
  DispatchKey highestPriorityTypeId() const;

  // Iteration
  DispatchKeySet iterator() const;
};

// Extract dispatch key from tensor inputs
DispatchKeySet dispatchKeySet(at::TensorList tensors);
```

---

## c10::SymInt, c10::SymFloat, c10::SymBool

These types support symbolic shapes in PyTorch, enabling shape inference and compilation without concrete dimension values. They wrap either a concrete value or a symbolic expression.

### Headers

```cpp
#include <c10/core/SymInt.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymBool.h>
```

### SymInt

```cpp
namespace c10 {

class SymInt {
public:
  // Constructors
  SymInt() : value_(0) {}
  SymInt(int64_t value);       // concrete value
  SymInt(SymNode value);       // symbolic value

  // Value access
  int64_t expect_int() const;  // crashes if symbolic
  bool is_symbolic() const noexcept;
  bool has_hint() const noexcept;

  // Operations
  SymInt operator+(const SymInt& other) const;
  SymInt operator-(const SymInt& other) const;
  SymInt operator*(const SymInt& other) const;
  SymInt operator/(const SymInt& other) const;
  SymInt operator%(const SymInt& other) const;

  // Comparison (returns SymBool for symbolic)
  SymBool operator==(const SymInt& other) const;
  SymBool operator!=(const SymInt& other) const;
  SymBool operator<(const SymInt& other) const;
  SymBool operator<=(const SymInt& other) const;
  SymBool operator>(const SymInt& other) const;
  SymBool operator>=(const SymInt& other) const;

  // Guards
  bool guard_int(const char* file, int64_t line) const;
  int64_t maybe_as_int() const;
};

} // namespace c10
```

### SymFloat

```cpp
namespace c10 {

class SymFloat {
public:
  SymFloat() : value_(0.0) {}
  SymFloat(double value);
  SymFloat(SymNode value);

  double expect_float() const;
  bool is_symbolic() const noexcept;

  SymFloat operator+(const SymFloat& other) const;
  SymFloat operator-(const SymFloat& other) const;
  SymFloat operator*(const SymFloat& other) const;
  SymFloat operator/(const SymFloat& other) const;
};

} // namespace c10
```

### SymBool

```cpp
namespace c10 {

class SymBool {
public:
  SymBool() : value_(false) {}
  SymBool(bool value);
  SymBool(SymNode value);

  bool expect_bool() const;
  bool is_symbolic() const noexcept;

  SymBool operator&&(const SymBool& other) const;
  SymBool operator||(const SymBool& other) const;
  SymBool operator!() const;
};

} // namespace c10
```

### Usage in Dynamic Shapes

```cpp
// SymInt enables dynamic shape compilation
// When compiling with torch.compile, tensor dimensions may be SymInt
// rather than concrete int64_t values

// Access symbolic shape
torch::Tensor t = torch::randn({3, 4});
// In compiled context, 3 and 4 might become SymInt values
c10::SymInt sym_dim = t.sym_size(0);

// Check if symbolic
sym_dim.is_symbolic();  // false in eager, may be true in compiled

// Operations on symbolic dimensions
c10::SymInt total = t.sym_size(0) * t.sym_size(1);
```

---

## c10::intrusive_ptr

`c10::intrusive_ptr` is a reference-counted smart pointer that is the backbone of PyTorch's object management. Unlike `std::shared_ptr`, the reference count is stored inside the object itself (intrusive), which provides better cache locality and allows the pointer to be retrieved from a raw pointer.

### Header

```cpp
#include <c10/util/intrusive_ptr.h>
```

### Class Definition

```cpp
namespace c10 {

template<typename T, typename Deleter = intrusive_ptr_target>
class intrusive_ptr {
public:
  // Constructors
  intrusive_ptr() noexcept = default;
  intrusive_ptr(T* ptr) noexcept;        // adopt reference
  intrusive_ptr(intrusive_ptr&& other) noexcept;
  intrusive_ptr(const intrusive_ptr& other);

  // Destructor (decrements refcount)
  ~intrusive_ptr();

  // Assignment
  intrusive_ptr& operator=(intrusive_ptr&& other) noexcept;
  intrusive_ptr& operator=(const intrusive_ptr& other);

  // Access
  T& operator*() const noexcept;
  T* operator->() const noexcept;
  T* get() const noexcept;

  // Modifiers
  void reset() noexcept;
  void reset(T* ptr) noexcept;

  // Use count
  int64_t use_count() const noexcept;

  // Boolean test
  explicit operator bool() const noexcept;
};

// Base class for intrusive-pointed objects
class intrusive_ptr_target {
public:
  intrusive_ptr_target() = default;
  virtual ~intrusive_ptr_target() = default;

  // Reference count access
  int64_t refcount() const noexcept;
  void incref() noexcept;
  void decref() noexcept;

private:
  std::atomic<size_t> refcount_{0};
};

// Factory function
template<typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args);

} // namespace c10
```

### Usage

```cpp
// PyTorch tensors are intrusive_ptr<TensorImpl>
// at::Tensor is essentially c10::intrusive_ptr<c10::TensorImpl>

// Custom intrusive-pointed class
class MyObject : public c10::intrusive_ptr_target {
public:
  int value;
  MyObject(int v) : value(v) {}
};

auto obj = c10::make_intrusive<MyObject>(42);
obj->value;  // 42
obj.refcount();  // 1

auto obj2 = obj;  // copy, refcount increases
obj.refcount();  // 2
obj2.refcount(); // 2
```

---

## c10/util/ Utilities

The `c10/util/` directory contains essential utility headers used throughout PyTorch.

### Exception Handling

```cpp
#include <c10/util/Exception.h>

// Error macros
#define TORCH_CHECK(condition, ...)     // assertion with message
#define TORCH_INTERNAL_ASSERT(cond, ...) // internal assertion
#define TORCH_CHECK_AT(cond, ...)       // ATen check

// Error class
namespace c10 {

class Error : public std::exception {
public:
  Error(const std::string& msg);
  Error(SourceLocation location, const std::string& msg);

  const char* what() const noexcept override;
  const std::string& msg() const;
  const SourceLocation& where() const;

  // Backtrace
  std::string backtrace() const;
};

class ValueError : public Error { /* ... */ };
class IndexError : public Error { /* ... */ };
class TypeError : public Error { /* ... */ };
class NotImplementedError : public Error { /* ... */ };

} // namespace c10
```

### Logging

```cpp
#include <c10/util/Logging.h>

// Logging macros (similar to glog)
#define C10_LOG(type) ...
#define VLOG(level) ...

// Log levels
enum class LogLevel {
  INFO,
  WARNING,
  ERROR,
  FATAL,
};

// Configuration
void SetLogLevel(LogLevel level);
void SetLoggingSink(std::shared_ptr<LoggingSink> sink);
```

### Math Compatibility

```cpp
#include <c10/util/math_compat.h>

// Provides math functions that work across platforms
// Includes: std::isfinite, std::isnan, std::isinf
// Portable implementations of ceil, floor, round, etc.
```

### irange

```cpp
#include <c10/util/irange.h>

// Python-like range for C++ loops
// Usage:
for (auto i : c10::irange(10)) {
  // i goes from 0 to 9
}

for (auto i : c10::irange(5, 10)) {
  // i goes from 5 to 9
}
```

### OptionalArrayRef

```cpp
#include <c10/util/OptionalArrayRef.h>

// Combination of optional and ArrayRef
using OptionalIntArrayRef = std::optional<c10::IntArrayRef>;

// Used for optional shape/dimension arguments
```

### Accumulator

```cpp
#include <c10/util/Accumulator.h>

// Thread-safe accumulator for profiling and metrics
namespace c10 {
class Accumulator {
public:
  void add(double value);
  double value() const;
  void reset();
};
}
```

### ConstexprCrc

```cpp
#include <c10/util/ConstexprCrc.h>

// Compile-time CRC computation for string hashing
// Used for dispatch key hashing
```

### Backports

```cpp
#include <c10/util/backports.h>

// Polyfills for C++14/17 features:
// - std::make_unique
// - std::optional
// - std::variant
// - fold expressions (for older compilers)
```

---

## c10/macros/

### Export.h

```cpp
#include <c10/macros/Export.h>

// Visibility and export macros
#define C10_API          // __attribute__((visibility("default")))
#define C10_HIDDEN       // __attribute__((visibility("hidden")))
#define TORCH_API
#define CAFFE2_API
#define CAFFE2_CORE_API

// Import/export for Windows DLL
#define C10_EXPORT
#define C10_IMPORT
```

### Macros.h

```cpp
#include <c10/macros/Macros.h>

// Compiler feature detection
#define C10_HAS_CPP_14
#define C10_HAS_CPP_17
#define C10_GCC_VERSION
#define C10_CLANG_VERSION
#define C10_MSVC_VERSION

// Platform detection
#define C10_ANDROID
#define C10_IOS
#define C10_MOBILE
#define C10_LINUX
#define C10_MACOS
#define C10_WINDOWS

// Compiler hints
#define C10_UNLIKELY(expr)
#define C10_LIKELY(expr)
#define C10_UNUSED
#define C10_ALWAYS_INLINE
#define C10_NOINLINE
#define C10_RESTRICT

// Alignment
#define C10_ALIGNAS(bytes)
#define C10_PACKED

// Deprecation
#define C10_DEPRECATED(message)
#define C10_DEPRECATED_MESSAGE(message)
```

---

## Key Header Files and Locations

### Core Types

| Header | Location | Description |
|--------|----------|-------------|
| `c10/core/Device.h` | Device type and index | Compute device representation |
| `c10/core/DeviceType.h` | DeviceType enum | All device type constants |
| `c10/core/ScalarType.h` | ScalarType enum | All data type constants |
| `c10/core/Scalar.h` | Scalar class | Multi-type scalar value |
| `c10/core/TensorImpl.h` | TensorImpl class | Core tensor implementation |
| `c10/core/Storage.h` | Storage class | Data buffer management |
| `c10/core/TensorOptions.h` | TensorOptions | Tensor creation options |
| `c10/core/DispatchKey.h` | DispatchKey enum | Dispatch system keys |
| `c10/core/DispatchKeySet.h` | DispatchKeySet | Set of dispatch keys |
| `c10/core/SymInt.h` | SymInt | Symbolic integer |
| `c10/core/SymFloat.h` | SymFloat | Symbolic float |
| `c10/core/SymBool.h` | SymBool | Symbolic boolean |
| `c10/core/ivalue.h` | IValue | Tagged union value type |
| `c10/core/Dict.h` | Dict | JIT dictionary type |
| `c10/core/List.h` | List | JIT list type |
| `c10/core/Layout.h` | Layout enum | Strided, Sparse, etc. |
| `c10/core/Allocator.h` | Allocator | Memory allocation interface |
| `c10/core/Backend.h` | Backend utilities | Backend-related helpers |

### Utility Headers

| Header | Description |
|--------|-------------|
| `c10/util/ArrayRef.h` | Non-owning array reference |
| `c10/util/SmallVector.h` | Small buffer optimized vector |
| `c10/util/OptionalRef.h` | Optional reference |
| `c10/util/Exception.h` | Error types and macros |
| `c10/util/Logging.h` | Logging infrastructure |
| `c10/util/irange.h` | Range-based for loop utility |
| `c10/util/intrusive_ptr.h` | Reference-counted smart pointer |
| `c10/util/math_compat.h` | Cross-platform math functions |
| `c10/util/ConstexprCrc.h` | Compile-time CRC |
| `c10/util/Accumulator.h` | Thread-safe accumulator |
| `c10/util/backports.h` | C++ feature polyfills |
| `c10/util/OptionalArrayRef.h` | Optional ArrayRef |
| `c10/util/ReadOnlyNoAliasDict.h` | Read-only dict for compilation |
| `c10/util/TypeSafeId.h` | Type-safe identifiers |
| `c10/util/strides.h` | Stride computation utilities |
| `c10/util/complex.h` | Complex number support |
| `c10/util/Half.h` | FP16 (half precision) type |
| `c10/util/BFloat16.h` | BF16 (bfloat16) type |
| `c10/util/Float8.h` | FP8 type variants |
| `c10/util/quantization.h` | Quantized type utilities |

### Macro Headers

| Header | Description |
|--------|-------------|
| `c10/macros/Export.h` | Visibility and export macros |
| `c10/macros/Macros.h` | Compiler and platform macros |
| `c10/macros/Mangle.h` | Name mangling utilities |

---

## Memory Management

c10 provides the `c10::Allocator` interface that backends implement for device-specific memory allocation:

```cpp
#include <c10/core/Allocator.h>

namespace c10 {

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual at::DataPtr allocate(size_t size) = 0;
  virtual void copy_data(void* dst, const void* src, size_t size) const;
};

// Global allocator access
Allocator* GetAllocator(DeviceType device_type);
void SetAllocator(DeviceType device_type, Allocator* allocator, uint8_t priority = 0);

// CUDA allocator (when CUDA available)
class CUDACachingAllocator : public Allocator {
  // Memory caching and pooling for CUDA
  // Implements CUDA memory management with caching
};

} // namespace c10
```

### DataPtr

```cpp
#include <c10/core/Allocator.h>

namespace c10 {

class DataPtr {
public:
  DataPtr() = default;
  DataPtr(void* data, void* ctx, DeleterFnPtr deleter, Device device);

  void* get() const noexcept;
  void* mutable_get() noexcept;
  Device device() const noexcept;
  void clear();

  // Move semantics only (no copy)
  DataPtr(DataPtr&& other) noexcept;
  DataPtr& operator=(DataPtr&& other) noexcept;
};

} // namespace c10
```

---

## Threading and Synchronization

c10 provides basic threading primitives:

```cpp
#include <c10/util/thread_name.h>

// Set/get thread names for debugging
void SetThreadName(const std::string& name);

#include <c10/util/Flags.h>

// Command-line flag parsing (for testing and debugging)
```

---

## Summary

The `c10` library is the bedrock of PyTorch's C++ infrastructure. Every tensor operation, dispatch decision, and data type choice ultimately depends on c10 types. Key takeaways:

1. **c10::TensorImpl** is the actual tensor object, accessed through `at::Tensor` (an intrusive pointer to TensorImpl)
2. **c10::DispatchKey** drives the dispatch system that selects the right kernel for each operation
3. **c10::IValue** is the universal value type for JIT and dispatcher arguments
4. **c10::intrusive_ptr** provides efficient reference counting used throughout PyTorch
5. **SymInt/SymFloat/SymBool** enable dynamic shape compilation in torch.compile
6. **ArrayRef/SmallVector** provide efficient array passing for tensor shapes and strides
