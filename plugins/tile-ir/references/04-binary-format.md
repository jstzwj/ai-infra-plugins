# Binary Format

The Tile IR bytecode is a binary representation of Tile IR modules including all items (globals, kernels, functions), attributes, and instructions. The bytecode format is stable, versioned, and provides Tile IR portability guarantees. The bytecode format produced by older Tile IR compilers/drivers can be read by newer compilers/drivers. A compiler/driver accepts bytecode up to its supported Tile IR version.

The remainder of this section describes the Tile IR bytecode format and its encoding.

The bytecode has a few top-level design goals:

- To represent a finite but expandable over time set of operations.
- To use a minimal set of types for the encoding.
- To enable manual construction and inspection by humans.
- To allow lazy loading of functions.
- To support forward and backward compatibility of the encoding.

The encoding of individual operations is described in Operations.

## Primitives

The bytecode is composed of a handful of primitive types, each with their own encoding described below. The format uses these primitives to encode more complex data structures for representing the full set of Tile IR items.

### Fixed-Width Integers

Fixed width integers are unsigned integers of a known size (in bytes). The values are stored in little-endian byte order.

> **Note:** All multi-byte values in the Tile IR bytecode format use little-endian encoding, including fixed-width integers, array offsets, and type indices.

```
byte ::= `0x00`...`0xFF`
```

Common fixed-width integer types used throughout the format:

| Type | Size | Description |
|------|------|-------------|
| `uint8_t` | 1 byte | 8-bit unsigned integer |
| `uint16_t` | 2 bytes | 16-bit unsigned integer (little-endian) |
| `uint32_t` | 4 bytes | 32-bit unsigned integer (little-endian) |
| `uint64_t` | 8 bytes | 64-bit unsigned integer (little-endian) |
| `int32_t` | 4 bytes | 32-bit signed integer (little-endian, two's complement) |
| `int64_t` | 8 bytes | 64-bit signed integer (little-endian, two's complement) |

**Example: Encoding uint32_t value 258**

```
Hex bytes: 0x02 0x01 0x00 0x00
// 258 = 0x00000102
// Little-endian: least significant byte first
```

**Example: Encoding uint64_t value 65537**

```
Hex bytes: 0x01 0x00 0x01 0x00 0x00 0x00 0x00 0x00
// 65537 = 0x0000000000010001
// Little-endian: least significant byte first
```

### Variable-Width Integers

Variable width integers, or VarInts, provide a compact representation for integers. Each encoded VarInt consists of one to nine bytes, which together represent a single 64-bit value. The Tile IR bytecode utilizes the PrefixVarInt encoding for VarInts. This encoding is a variant of the LEB128 ("Little-Endian Base 128") encoding, where each byte of the encoding provides up to 7 bits for the value, with the remaining bit used to store a tag indicating the number of bytes used for the encoding. Small unsigned integers (less than 2^7) may be stored in one byte, larger unsigned integers (up to 2^14) may be stored in two bytes, and so on.

```
varint ::= `0x00`...`0xFF`
```

**PrefixVarInt Encoding Detail**

The PrefixVarInt encoding uses a prefix tag in the first byte to signal how many bytes follow:

| First byte range | Prefix bits | Total bytes | Value range |
|-----------------|-------------|-------------|-------------|
| `0x00`-`0x7F` | 0 (high bit = 0) | 1 | 0 to 127 |
| `0x80`-`0xBF` | `10` (top 2 bits) | 2 | 0 to 16,383 |
| `0xC0`-`0xDF` | `110` (top 3 bits) | 3 | 0 to 2,097,151 |
| `0xE0`-`0xEF` | `1110` (top 4 bits) | 4 | 0 to 268,435,455 |
| `0xF0`-`0xF7` | `11110` (top 5 bits) | 5 | 0 to 34,359,738,367 |
| `0xF8`-`0xFB` | `111110` (top 6 bits) | 6 | larger values |
| `0xFC`-`0xFD` | `1111110` (top 7 bits) | 7 | larger values |
| `0xFE` | `11111110` (top 8 bits) | 8 | larger values |
| `0xFF` | `11111111` (all bits) | 9 | full 64-bit range |

**Examples:**

```
Value 0:    [0x00]                   // 1 byte
Value 42:   [0x2A]                   // 1 byte (0x2A = 42)
Value 127:  [0x7F]                   // 1 byte (max single-byte)
Value 128:  [0x80, 0x01]            // 2 bytes (0x80 prefix = 2-byte, data = 0x01 = 1, (1 << 7) | ... = 128)
Value 300:  [0x80 | ((300 >> 0) & 0x3F), (300 >> 6) & 0xFF]  // 2 bytes
Value 16383: [0xBF, 0xFF]           // 2 bytes (max two-byte)
```

## File Structure

The Tile IR bytecode file is composed of a stream of bytes; which is composed of a header followed by multiple sections. The header contains the 8-byte "magic number", and a version number.

Each section is expected to appear once within a bytecode file and is identified by a type code, the length, alignment, padding, and the payload. This design allows forward compatibility and selective parsing (tools can skip unknown sections, or unneeded sections).

The overall file layout is shown below:

```
+================================================================+
|                        Tile IR Bytecode                         |
+================================================================+
|                                                                |
|  +----------------------------------------------------------+  |
|  |                    Header                                  |  |
|  |  Magic: 0x7F 'T' 'i' 'l' 'e' 'I' 'R' 0x00   (8 bytes)    |  |
|  |  Version: major(uint8) minor(uint8) tag(uint16) (4 bytes)  |  |
|  +----------------------------------------------------------+  |
|                                                                |
|  +----------------------------------------------------------+  |
|  |                  Section 1 (String)                        |  |
|  |  idAndIsAligned: byte                                      |  |
|  |  length: varint                                            |  |
|  |  [alignment: varint]                                       |  |
|  |  [padding: 0xCB bytes]                                     |  |
|  |  data: byte[]                                              |  |
|  +----------------------------------------------------------+  |
|                                                                |
|  +----------------------------------------------------------+  |
|  |                  Section 2 (Function Table)                |  |
|  |  ...                                                       |  |
|  +----------------------------------------------------------+  |
|                                                                |
|  |            ... additional sections ...                     |  |
|                                                                |
|  +----------------------------------------------------------+  |
|  |              End-of-Bytecode Marker (0x00)                 |  |
|  +----------------------------------------------------------+  |
|                                                                |
+================================================================+
```

```
bytecode {
  magic: "\x7FTileIR\x00",
  version: varint,
  sections: section[]
}
```

### Magic Number

The Tile IR bytecode magic number consumes 8 bytes and is `\x7FTileIR\x00`. The magic number must be present at the beginning of the bytecode file to be accepted by the driver.

```
Magic bytes (hex):  7F 54 69 6C 65 49 52 00
Magic bytes (char): \x7F T  i  l  e  I  R  \x00
```

The magic number serves several purposes:
- File format identification for tools and operating systems
- Quick validation that a file is a valid Tile IR bytecode file
- The leading `0x7F` byte ensures the file is not mistaken for a text file

### Version

Following the magic number is the version of the bytecode format. The version allows both forwards and backwards compatibility of the format.

The version is encoded as three little-endian fixed-width fields immediately following the magic number:

```
version {
  major: uint8_t,    // Major version number
  minor: uint8_t,    // Minor version number
  tag: uint16_t      // Version tag (little-endian)
}
```

**Version Field Detail:**

| Field | Type | Offset | Description |
|-------|------|--------|-------------|
| `major` | `uint8_t` | 8 | Major version number. Incremented for breaking changes. |
| `minor` | `uint8_t` | 9 | Minor version number. Incremented for backward-compatible additions. |
| `tag` | `uint16_t` | 10 | Version tag for distinguishing builds within a version. |

The total header size is 12 bytes (8 bytes magic + 4 bytes version).

Each file encodes the version which allows parsers to adapt to the specific version. We do not break old versions of the format, if a newer file uses features an older parser can't interpret, it skips unknown optional features but will fail gracefully in cases where new features are unsupported. A new producer can also target older versions of the bytecode directly ensuring compatibility with older drivers.

**Version Compatibility Guarantees**

Backward Compatibility (a new deserializer can read old bytecode):

- The deserializer must support all previous versions of the bytecode format.
- The deserializer must handle all previously existing opcodes, types, and sections.
- The deserializer must maintain original program semantics.

Forward Compatibility (an old deserializer can read new bytecode):

- The deserializer must fail only if it encounters unknown required features.
- The deserializer must only read bytecode up to its version.
- The deserializer must error with a clear message if there is a version mismatch.

Version Targeting (a serializer can target specific older versions):

- The serializer may target one or more specific older versions.
- The serializer must validate all features are supported by the requested target version.
- The serializer must fail if attempting to serialize unsupported features.

Optional vs Required Features (required changes are clearly documented):

- Required changes must be introduced in a new bytecode version and graceful failure must be designed.
- Optional changes must be clearly documented and must preserve the above guarantees.

Examples:

- For new ops: Add new opcodes
- For changes to existing ops: Typically keep the old format for backward compatibility and add new opcode rather than modifying existing ones

**Version Compatibility Summary Table:**

| Guarantee Type | Rules |
|---|---|
| **Backward Compatibility** | Deserializer must support all previous versions; handle all existing opcodes, types, sections; maintain original semantics |
| **Forward Compatibility** | Deserializer must fail only on unknown required features; only read up to its version; error with clear message on version mismatch |
| **Version Targeting** | Serializer may target older versions; must validate features supported by target; must fail on unsupported features |
| **Optional vs Required** | Required changes need new bytecode version with graceful failure; Optional changes must preserve guarantees |

## Sections

The remainder of the bytecode stream is composed of sections. Sections are used to group data within the bytecode and allow operations on the stream to be per-section enabling out-of-order processing of data and/or lazy-loading. Each section contains a section ID, whose high bit indicates if the section has alignment requirements, a length and an optional alignment. A section ID is a 7-bit integer with the high bit indicating if the section has alignment requirements. When an alignment is present, a variable number of padding bytes (each byte = 0xCB) may appear before the section data. The alignment of a section must be a power of 2. The alignment is represented as a VarInt. The padding ensures the section data starts at the specified alignment boundary.

```
section {
  idAndIsAligned: byte   // low 7 bits = section ID, high bit = alignment bit
  length: varint,
  alignment: varint?,    // present only if high bit was set
  padding: byte[],       // bytes of 0xCB as needed
  data: byte[]
}
```

**Section Header Byte Encoding:**

```
idAndIsAligned byte:
+---+---+---+---+---+---+---+---+
| A |    Section ID (0-127)      |
+---+---+---+---+---+---+---+---+
  7                              0

A = Alignment bit
  0 = No alignment required (alignment field absent)
  1 = Alignment required (alignment field present)

Section ID values:
  0x01 = String Section
  0x02 = Function Table Section
  0x03 = Debug Section
  0x04 = Constant Data Section
  0x05 = Type Section
  0x06 = Global Section
```

**Example: Section with alignment**

```
idAndIsAligned = 0x85       // High bit set (aligned), section ID = 0x05 (Type)
                             // Binary: 10000101
alignment = 8               // 8-byte alignment
padding = 0xCB 0xCB 0xCB    // Padding to reach next 8-byte boundary
data = ...                  // Type section data
```

**Example: Section without alignment**

```
idAndIsAligned = 0x01       // High bit clear (no alignment), section ID = 0x01 (String)
                             // Binary: 00000001
length = 256                // 256 bytes of string data follow
data = ...                  // String section data (no alignment field, no padding)
```

**Deserialization**

The following are the implications of the section format on the deserialization process.

- The deserializer must read the section ID and length (idAndIsAligned) as a single byte.
- If the section ID has the high bit set, the deserializer must read the alignment (alignment) and apply the alignment by skipping 0xCB bytes as needed.
- The deserializer must read the length (length) bytes of the payload. The length is a VarInt.
- The deserializer must move on to the next section until EOF (end of file).

### String Section

The string section holds all textual names used by the module to avoid repeating them inline. The section is encoded with the total number of strings, followed by the start index of each of the individual strings. The remaining encoding contains a single blob containing all the strings concatenated together. This design allows loading a specific string without reading the whole section. Strings in the bytecode are stored as raw byte sequences (in UTF-8 encoding) with an associated size. Finding the i-th string involves jumping to stringStartIndex[i] and reading until the next offset or end of the blob.

```
strings {
  numStrings: varint,
  stringStartIndex: uint32[],
  stringData: byte[]
}
```

**String Section Layout:**

```
+---------------------------------------------------+
| String Section                                     |
+---------------------------------------------------+
| numStrings: varint     (e.g., 3 strings)          |
+---------------------------------------------------+
| stringStartIndex[0]: uint32_t (offset to string 0)|
| stringStartIndex[1]: uint32_t (offset to string 1)|
| stringStartIndex[2]: uint32_t (offset to string 2)|
+---------------------------------------------------+
| stringData blob:                                   |
|  "kernel_A\0" "threadIdx\0" "blockDim\0"         |
+---------------------------------------------------+
```

**String Access Algorithm:**

```
function getString(index):
    start = stringStartIndex[index]
    if index + 1 < numStrings:
        end = stringStartIndex[index + 1]
    else:
        end = len(stringData)
    return stringData[start:end]
```

**Example:**

```
numStrings = 3
stringStartIndex = [0, 9, 19]
stringData = "kernel_A\0threadIdx\0blockDim\0"

String 0: "kernel_A"   (offset 0..8)
String 1: "threadIdx"  (offset 9..18)
String 2: "blockDim"   (offset 19..27)
```

### Function Table Section

The function table section enumerates the module's functions and embeds their code inline. First, numFunctions indicates how many functions follow; for each function i, nameIndex[i] is an index into the string section naming the function and signatureIndex[i] is an index into the Type section specifying its parameter, return types, global flags, etc (so multiple functions with identical signatures can share the same type entry). The field functionLocIndex[i] is a VarInt referencing an entry in the Debug Section that describes the function's definition location (e.g., source file and line). If functionLocIndex[i] is zero, there is no associated debug metadata for the function's definition scope. lengthOfFunction[i] states how many bytes of code belong to function i, and functionBody[i] contains exactly those bytes of instruction encodings. This layout avoids a separate code section and makes parsing each function straightforward: once you read the metadata for function i, you can either parse its instructions directly or skip them by advancing lengthOfFunction[i] bytes. The instruction encoding itself allows each operation to include a slot for its source location, referencing an entry in the debug section.

The instruction encodings (opcodes, operands, etc.) are described later under Operation Opcodes and Encodings Section.

The function table encoding has been updated to include function flags and optional optimization hints:

```
functionTable {
  numFunctions : varint
  // for each function i in [0..numFunctions-1]:
  nameIndex[i] : varint                    // References the function's name in the StringSec
  signatureIndex[i] : varint               // References the extended function signature in the TypeSec
  entryFlag[i] : byte                      // Function flags (visibility, kind, optimization hints)
  functionLocIndex[i] : varint             // 0 means no debug location
  optimizationHints[i]? : self-contained   // Present only if HasOptimizationHints flag is set
  lengthOfFunction[i] : varint
  functionBody[i] : byte[lengthOfFunction[i]]
}
```

The entryFlag byte encodes the following information:

| Bit | Mask | Field | Meaning |
|-----|------|-------|---------|
| 0 | 0x01 | Visibility | 0 = Public, 1 = Private |
| 1 | 0x02 | Function Kind | 0 = Device Function, 1 = Kernel Entry Point |
| 2 | 0x04 | Optimization Hints | 0 = No hints present, 1 = Optimization hints follow |
| 3-7 | - | Reserved | Reserved for future extensions |

**entryFlag byte layout:**

```
entryFlag byte:
+---+---+---+---+---+---+---+---+
| R | R | R | R | R | O | K | V |
+---+---+---+---+---+---+---+---+
  7   6   5   4   3   2   1   0

V = Visibility (0=Public, 1=Private)
K = Kind (0=Device, 1=Kernel)
O = Has Optimization Hints (0=No, 1=Yes)
R = Reserved
```

**Example: Encoding a public kernel entry point with optimization hints**

```
entryFlag = 0x06   // Binary: 00000110
                   // V=0 (Public), K=1 (Kernel), O=1 (Has Hints)

entryFlag = 0x01   // Binary: 00000001
                   // V=1 (Private), K=0 (Device Function), O=0 (No Hints)
```

**Function Table Layout Diagram:**

```
+-----------------------------------------------------------------+
| Function Table Section                                           |
+-----------------------------------------------------------------+
| numFunctions: varint (e.g., 2)                                  |
+-----------------------------------------------------------------+
| Function 0:                                                     |
|   nameIndex[0]: varint         -> "matmul_kernel"               |
|   signatureIndex[0]: varint    -> FunctionType(Tensor, Tensor)  |
|   entryFlag[0]: 0x02           -> Public Kernel Entry            |
|   functionLocIndex[0]: varint  -> debug offset or 0             |
|   lengthOfFunction[0]: varint  -> e.g., 256                     |
|   functionBody[0]: byte[256]   -> instruction bytes             |
+-----------------------------------------------------------------+
| Function 1:                                                     |
|   nameIndex[1]: varint         -> "helper_func"                  |
|   signatureIndex[1]: varint    -> FunctionType()                 |
|   entryFlag[1]: 0x01           -> Private Device Function        |
|   functionLocIndex[1]: 0       -> no debug info                  |
|   lengthOfFunction[1]: varint  -> e.g., 128                     |
|   functionBody[1]: byte[128]   -> instruction bytes             |
+-----------------------------------------------------------------+
```

### Constant Data Section

The constant data section holds large constants (e.g., dense tensors, large arrays) separately from code.

The current implementation does not impose explicit size limits on individual constants. Constants are stored using uint64_t offsets, allowing for very large constant data. However, practical limits may be imposed by available memory and any downstream compiler or runtime limitations (such as CUBIN generation constraints).

As we have done with the string section we move the constants into their own section to avoid bloating the function table section and allow for the lazy loading of specific constants when needed. Individual operations may reference constants by an index into this section. The section is encoded with the total number of constants, followed by the start index of each of the individual constants. The remaining encoding contains a single blob containing all the constants concatenated together.

```
constant {
  numConstants: varint,
  constantStartIndex: uint64_t[],
  constantData: byte[]
}
```

**Constant Section Layout Diagram:**

```
+---------------------------------------------------+
| Constant Data Section                              |
+---------------------------------------------------+
| numConstants: varint (e.g., 2)                    |
+---------------------------------------------------+
| constantStartIndex[0]: uint64_t  -> offset 0      |
| constantStartIndex[1]: uint64_t  -> offset 128    |
+---------------------------------------------------+
| constantData:                                      |
|   [0..127]:   Dense FP16 tensor (128 bytes)        |
|   [128..255]: Dense I32 array (128 bytes)          |
+---------------------------------------------------+
```

**Constant Access Algorithm:**

```
function getConstant(index):
    start = constantStartIndex[index]
    if index + 1 < numConstants:
        end = constantStartIndex[index + 1]
    else:
        end = len(constantData)
    return constantData[start:end]
```

### Type Section

The type section stores all type definitions (scalar types, function signatures, parametric types, etc.) used in the module. The section begins with numTypes specifying how many types follow, then an array of offsets (typeStartIndex) into the encoded blob (typeData). To load the i-th type definition, you use the offset contained in typeStartIndex[i] to index into the typeData blob and parse the definition from there. The typeData blob is a single encoded binary blob containing the type definitions concatenated together.

```
type {
  numTypes: varint
  typeStartIndex: uint32_t[] // array of offsets, length = numTypes
  typeData: byte[]           // concatenated bytes for all type definitions
}
```

**Type Section Layout Diagram:**

```
+---------------------------------------------------+
| Type Section                                       |
+---------------------------------------------------+
| numTypes: varint (e.g., 5)                        |
+---------------------------------------------------+
| typeStartIndex[0]: uint32_t -> offset 0            |
| typeStartIndex[1]: uint32_t -> offset 1            |
| typeStartIndex[2]: uint32_t -> offset 2            |
| typeStartIndex[3]: uint32_t -> offset 10           |
| typeStartIndex[4]: uint32_t -> offset 25           |
+---------------------------------------------------+
| typeData:                                          |
|   [0]: typeTag=0x03 (i32)                         |
|   [1]: typeTag=0x07 (f32)                         |
|   [2..9]: typeTag=0x0D (Tile), elementType=1,     |
|           shape=[128, 64]                          |
|   [10..24]: typeTag=0x10 (Function), ...           |
|   [25]: typeTag=0x11 (Token)                       |
+---------------------------------------------------+
```

Each individual type definition will consist of a typeTag followed by a predefined payload structure specific to that typeTag. This deterministic approach allows parsers to understand the payload layout based solely on the typeTag.

```
typeTag : byte
payload : byte[lengthOfPayload]
```

typeTag indicates the "kind" of type. This can be a simple scalar, a function signature, or a more complex structure like a tensor.

payload is interpreted based on typeTag. For instance, a function signature might store the number of parameters, references to their types, etc, while a tensor type might store dimensions and an element type.

### Detailed Type Encodings

The following sections describe the specific encoding format for each type tag.

#### Integer Types (typeTag = 0x00-0x04)

Integer types require no additional payload data beyond the type tag:

```c
// Complete encoding: 1 byte total
I1:   typeTag = 0x00  // 1-bit boolean
I8:   typeTag = 0x01  // 8-bit integer
I16:  typeTag = 0x02  // 16-bit integer
I32:  typeTag = 0x03  // 32-bit integer
I64:  typeTag = 0x04  // 64-bit integer
```

| typeTag | Meaning | Size | Payload Size | Total Encoding |
|---------|---------|------|-------------|----------------|
| 0x00 | i1 (boolean) | 1 bit | 0 bytes | 1 byte |
| 0x01 | i8 | 8 bits | 0 bytes | 1 byte |
| 0x02 | i16 | 16 bits | 0 bytes | 1 byte |
| 0x03 | i32 | 32 bits | 0 bytes | 1 byte |
| 0x04 | i64 | 64 bits | 0 bytes | 1 byte |

**Example: Encoding an i32 type**

```
Hex bytes: 0x03
```

#### Float Types (typeTag = 0x05-0x0B)

Float types require no additional payload data beyond the type tag:

```c
// Complete encoding: 1 byte total
F16:      typeTag = 0x05  // 16-bit IEEE float (half precision)
BF16:     typeTag = 0x06  // 16-bit bfloat (brain float)
F32:      typeTag = 0x07  // 32-bit IEEE float (single precision)
TF32:     typeTag = 0x08  // TensorFloat-32 (19-bit mantissa)
F64:      typeTag = 0x09  // 64-bit IEEE float (double precision)
F8E4M3FN: typeTag = 0x0A  // 8-bit float (E4M3FN format, FP8)
F8E5M2:   typeTag = 0x0B  // 8-bit float (E5M2 format, FP8)
```

| typeTag | Meaning | Bit Width | Exponent Bits | Mantissa Bits | Payload |
|---------|---------|-----------|---------------|---------------|---------|
| 0x05 | F16 | 16 | 5 | 10 | None |
| 0x06 | BF16 | 16 | 8 | 7 | None |
| 0x07 | F32 | 32 | 8 | 23 | None |
| 0x08 | TF32 | 19 | 8 | 10 | None |
| 0x09 | F64 | 64 | 11 | 52 | None |
| 0x0A | F8E4M3FN | 8 | 4 | 3 | None |
| 0x0B | F8E5M2 | 8 | 5 | 2 | None |

**Example: Encoding an F32 type**

```
Hex bytes: 0x07
```

#### Pointer Type (typeTag = 0x0C)

```c
typeTag : byte = 0x0C      // Pointer type
pointeeTypeIndex : varint  // Index of the pointee type
```

**Example: Encoding a pointer to i32 (assuming i32 is type index 0)**

```
Hex bytes: 0x0C 0x00
// typeTag=0x0C (Pointer), pointeeTypeIndex=0 (i32)
```

**Example: Encoding a pointer to F16 (assuming F16 is type index 5)**

```
Hex bytes: 0x0C 0x05
// typeTag=0x0C (Pointer), pointeeTypeIndex=5 (F16)
```

#### Tile Type (typeTag = 0x0D)

```c
typeTag : byte = 0x0D      // Tile type
elementTypeIndex : varint  // Index of the element type
shape : int64_t[]          // Shape dimensions (var-length array)
```

The shape is encoded as a varint count followed by the dimensions:

```
numDims: varint            // Number of dimensions
dims: int64_t[numDims]     // Dimension values as int64_t
```

**Example: Encoding a 128x64 F32 tile**

```
Hex bytes:
  0x0D                   // typeTag = Tile
  0x07                   // elementTypeIndex = 7 (F32)
  0x02                   // numDims = 2
  0x80 0x01              // dim[0] = 128 (varint encoded)
  0x40                   // dim[1] = 64  (varint encoded)
```

#### TensorView Type (typeTag = 0x0E)

```c
typeTag : byte = 0x0E      // TensorView type
elementTypeIndex : varint  // Index of the element type
numDims : varint           // Number of dimensions
shape : int64_t[numDims]   // Shape dimensions
strides : int64_t[numDims] // Stride values
indexTypeTag : byte        // Index type (I32=0x03 or I64=0x04)
```

**Example: Encoding a 2D TensorView of BF16 with strides**

```
Hex bytes:
  0x0E                   // typeTag = TensorView
  0x06                   // elementTypeIndex = 6 (BF16)
  0x02                   // numDims = 2
  [shape: 256, 128]      // shape dimensions as int64_t pairs
  [strides: 128, 1]      // stride values as int64_t pairs
  0x03                   // indexTypeTag = I32
```

#### PartitionView Type (typeTag = 0x0F)

```c
typeTag : byte = 0x0F        // PartitionView type
numTileDims : varint         // Number of tile shape dimensions
tileShape : int32_t[numTileDims]   // Tile shape
tensorViewTypeIndex : varint // Index of the TensorView type
numDimMap : varint           // Number of dimension mapping entries
dimMap : int32_t[numDimMap]  // Dimension mapping
masked : byte                // Masked flag (0x00=false, 0x01=true)
```

**Example: Encoding a 2D partition view**

```
Hex bytes:
  0x0F                   // typeTag = PartitionView
  0x02                   // numTileDims = 2
  [32, 64 as int32_t]    // tileShape = [32, 64]
  0x03                   // tensorViewTypeIndex = 3
  0x02                   // numDimMap = 2
  [0, 1 as int32_t]      // dimMap = [0, 1]
  0x00                   // masked = false
```

#### Function Type (typeTag = 0x10)

```c
typeTag : byte = 0x10 // Function Type
numParams : varint    // Number of input parameters
paramTypeIndices : varint[]  // Array of type indices for each parameter
numResults : varint   // Number of results
resultTypeIndices : varint[] // Array of type indices for each return value
```

The function type encoding stores only the essential type information (input and result types). Argument attributes and other function metadata are stored separately in the function table section.

**Example: Encoding a function type (i32, F32) -> (F32)**

```
Hex bytes:
  0x10                   // typeTag = Function
  0x02                   // numParams = 2
  0x03                   // paramTypeIndices[0] = 3 (i32)
  0x07                   // paramTypeIndices[1] = 7 (F32)
  0x01                   // numResults = 1
  0x07                   // resultTypeIndices[0] = 7 (F32)
```

#### Token Type (typeTag = 0x11)

```c
typeTag : byte = 0x11  // Token type (no additional payload)
```

**Example: Encoding a token type**

```
Hex bytes: 0x11
```

#### Complete Type Tag Reference

| typeTag | Type | Payload Fields | Payload Size |
|---------|------|---------------|-------------|
| 0x00 | i1 | None | 0 |
| 0x01 | i8 | None | 0 |
| 0x02 | i16 | None | 0 |
| 0x03 | i32 | None | 0 |
| 0x04 | i64 | None | 0 |
| 0x05 | F16 | None | 0 |
| 0x06 | BF16 | None | 0 |
| 0x07 | F32 | None | 0 |
| 0x08 | TF32 | None | 0 |
| 0x09 | F64 | None | 0 |
| 0x0A | F8E4M3FN | None | 0 |
| 0x0B | F8E5M2 | None | 0 |
| 0x0C | Pointer | pointeeTypeIndex : varint | variable |
| 0x0D | Tile | elementTypeIndex : varint, shape : int64_t[] | variable |
| 0x0E | TensorView | elementTypeIndex : varint, shape : int64_t[], strides : int64_t[], indexTypeTag : byte | variable |
| 0x0F | PartitionView | tileShape : int32_t[], tensorViewTypeIndex : varint, dimMap : int32_t[], masked : byte | variable |
| 0x10 | Function | numParams : varint, paramTypeIndices : varint[], numResults : varint, resultTypeIndices : varint[] | variable |
| 0x11 | Token | None | 0 |

### Attribute Encoding

Attributes in Tile IR bytecode can be encoded in two ways:

- **Inline encoding** - Simple attributes are encoded directly in the instruction stream
- **Self-contained encoding** - Complex attributes include a type tag followed by their data

Self-contained attributes use the following format:

```
attributeTag : byte
attributeData : byte[]  // Format depends on attributeTag
```

The following attribute tags are supported:

#### Integer Attribute (attributeTag = 0x01)

```c
attributeTag : byte = 0x01  // Integer attribute
typeIndex : varint          // Type index for the integer type
value : varint              // Integer value (zero-extended)
```

**Example: Encoding integer attribute value 42 of type i32 (type index 3)**

```
Hex bytes: 0x01 0x03 0x2A
// attributeTag=0x01 (Integer), typeIndex=3 (i32), value=42
```

**Example: Encoding integer attribute value 1000 of type i64 (type index 4)**

```
Hex bytes: 0x01 0x04 0xE8 0x07
// attributeTag=0x01, typeIndex=4 (i64), value=1000 (varint 0xE8 0x07)
```

#### Float Attribute (attributeTag = 0x02)

```c
attributeTag : byte = 0x02  // Float attribute
typeIndex : varint          // Type index for the float type
value : byte[]              // APFloat representation (variable length)
```

The APFloat representation stores the floating-point value in its native binary format. The size depends on the float type:
- F16: 2 bytes
- BF16: 2 bytes
- F32: 4 bytes (IEEE 754 binary32)
- F64: 8 bytes (IEEE 754 binary64)
- F8E4M3FN: 1 byte
- F8E5M2: 1 byte

**Example: Encoding F32 value 3.14 (type index 7, approximately 0x4048F5C3)**

```
Hex bytes: 0x02 0x07 0xC3 0xF5 0x48 0x40
// attributeTag=0x02, typeIndex=7 (F32), value=3.14 as IEEE 754 LE
```

#### Bool Attribute (attributeTag = 0x03)

```c
attributeTag : byte = 0x03  // Bool attribute
value : byte                // 0x00=false, 0x01=true
```

**Example: Encoding boolean true**

```
Hex bytes: 0x03 0x01
// attributeTag=0x03, value=true
```

#### Type Attribute (attributeTag = 0x04)

```c
attributeTag : byte = 0x04  // Type attribute
typeIndex : varint          // Index of the referenced type
```

**Example: Encoding a reference to F16 (type index 5)**

```
Hex bytes: 0x04 0x05
// attributeTag=0x04, typeIndex=5 (F16)
```

#### String Attribute (attributeTag = 0x05)

```c
attributeTag : byte = 0x05  // String attribute
stringIndex : varint        // Index into the string section
```

**Example: Encoding a reference to string at index 3**

```
Hex bytes: 0x05 0x03
// attributeTag=0x05, stringIndex=3
```

#### Array Attribute (attributeTag = 0x06)

```c
attributeTag : byte = 0x06    // Array attribute
numElements : varint          // Number of elements
elements : self-contained[]   // Array of self-contained attributes
```

**Example: Encoding an array of two integers [10, 20]**

```
Hex bytes: 0x06 0x02 0x01 0x03 0x0A 0x01 0x03 0x14
// attributeTag=0x06 (Array), numElements=2
//   element[0]: 0x01 (Integer), typeIndex=3 (i32), value=10
//   element[1]: 0x01 (Integer), typeIndex=3 (i32), value=20
```

#### DenseElements Attribute (attributeTag = 0x07)

```c
attributeTag : byte = 0x07  // DenseElements attribute
typeIndex : varint          // Type index for the shaped type
constantIndex : varint      // Index into constant section (for numeric data)
// OR for string data:
numStrings : varint         // Number of string elements
stringIndices : varint[]    // Indices into string section
```

The DenseElements attribute has two variants depending on whether the data is numeric or string:

**Numeric variant:**

```
0x07 typeIndex constantIndex
```

**String variant:**

```
0x07 typeIndex numStrings stringIndices[]
```

**Example: Encoding a dense F32 tensor referencing constant data**

```
Hex bytes: 0x07 0x07 0x00
// attributeTag=0x07, typeIndex=7 (F32 tensor), constantIndex=0
```

#### DivBy Attribute (attributeTag = 0x08)

```c
attributeTag : byte = 0x08    // DivBy attribute
divisor : varint              // Divisor value
flags : byte                  // Bit 0: unsignedInt, Bit 1: hasEvery, Bit 2: hasAlong
every : signed_varint?        // Present if Bit 1 set
along : signed_varint?        // Present if Bit 2 set
```

**DivBy flags byte layout:**

```
flags byte:
+---+---+---+---+---+---+---+---+
| R | R | R | R | R | A | E | U |
+---+---+---+---+---+---+---+---+
  7   6   5   4   3   2   1   0

U = unsignedInt (bit 0)
E = hasEvery (bit 1)
A = hasAlong (bit 2)
R = Reserved
```

**Example: Encoding DivBy with divisor=16, unsigned, with every=4**

```
Hex bytes: 0x08 0x10 0x03 0x04
// attributeTag=0x08, divisor=16, flags=0x03 (unsigned + hasEvery), every=4
```

#### SameElements Attribute (attributeTag = 0x09)

```c
attributeTag : byte = 0x09  // SameElements attribute
numValues : varint          // Number of values
values : int64_t[]          // Array of int64 values
```

**Example: Encoding SameElements with values [1, 2, 3]**

```
Hex bytes: 0x09 0x03 [1 as int64_t] [2 as int64_t] [3 as int64_t]
// attributeTag=0x09, numValues=3, values=[1, 2, 3]
```

#### Dictionary Attribute (attributeTag = 0x0A)

```c
attributeTag : byte = 0x0A      // Dictionary attribute
numEntries : varint             // Number of key-value pairs
entries : dictEntry[]           // Array of dictionary entries
```

```
dictEntry {
  keyStringIndex : varint       // Index of key string
  value : self-contained        // Self-contained attribute value
}
```

**Example: Encoding a dictionary with one entry {"fast" = true}**

```
Hex bytes: 0x0A 0x01 0x00 0x03 0x01
// attributeTag=0x0A, numEntries=1
//   entry[0]: keyStringIndex=0 ("fast"), value=Bool(true)
```

#### OptimizationHints Attribute (attributeTag = 0x0B)

```c
attributeTag : byte = 0x0B  // OptimizationHints attribute
dictionary : dictionary      // Dictionary attribute (without tag)
```

The OptimizationHints attribute contains an inline dictionary (without the dictionary attribute tag byte). This is used to provide optimization hints to the compiler such as unrolling factors, tiling strategies, or memory access patterns.

#### NonNegative Attribute (attributeTag = 0x0C)

```c
attributeTag : byte = 0x0C  // NonNegative attribute
// No additional payload - presence indicates non-negative constraint
```

**Complete Attribute Tag Reference:**

| Tag | Attribute | Payload Fields |
|-----|-----------|---------------|
| 0x01 | Integer | typeIndex : varint, value : varint |
| 0x02 | Float | typeIndex : varint, value : byte[] |
| 0x03 | Bool | value : byte (0x00/0x01) |
| 0x04 | Type | typeIndex : varint |
| 0x05 | String | stringIndex : varint |
| 0x06 | Array | numElements : varint, elements : self-contained[] |
| 0x07 | DenseElements | typeIndex : varint, constantIndex : varint (numeric) OR numStrings : varint, stringIndices : varint[] (string) |
| 0x08 | DivBy | divisor : varint, flags : byte, every? : signed_varint, along? : signed_varint |
| 0x09 | SameElements | numValues : varint, values : int64_t[] |
| 0x0A | Dictionary | numEntries : varint, entries : dictEntry[] |
| 0x0B | OptimizationHints | dictionary : dictionary (inline, no tag) |
| 0x0C | NonNegative | No payload |

### Global Section

The global section stores module-level global variables. This section is optional and is only present if the module contains cuda_tile.global operations.

```
global {
  numGlobals: varint
  // for each global i in [0..numGlobals-1]:
  symbolNameIndex[i] : varint   // References the global's symbol name in the StringSec
  valueTypeIndex[i] : varint    // Type index for the global's value type
  constantValueIndex[i] : varint // Index into constant section for the global's initial value
}
```

Each global variable is encoded with:

- **symbolNameIndex:** Index into the string section for the global's symbol name
- **valueTypeIndex:** Index into the type section for the global's type (typically a shaped type like tensor)
- **constantValueIndex:** Index into the constant section containing the global's initial value

**Global Section Layout Diagram:**

```
+---------------------------------------------------+
| Global Section                                     |
+---------------------------------------------------+
| numGlobals: varint (e.g., 2)                      |
+---------------------------------------------------+
| Global 0:                                          |
|   symbolNameIndex[0]: varint -> "global_buffer"   |
|   valueTypeIndex[0]: varint -> TensorType(256,F32)|
|   constantValueIndex[0]: varint -> constant[0]     |
+---------------------------------------------------+
| Global 1:                                          |
|   symbolNameIndex[1]: varint -> "global_mask"      |
|   valueTypeIndex[1]: varint -> TensorType(128,I1) |
|   constantValueIndex[1]: varint -> constant[1]     |
+---------------------------------------------------+
```

### Debug Section

The debug section stores the serialized debug information (for more details about debug information see Debug Info). This section is optional as certain tools may ignore it and serializers may leave it empty for release builds.

```
debug {
  diOpsNum: varint          // Total number of operations with debug info
  padding: bytes            // Align to 4 bytes
  diIndexOffsets: uint32_t[] // Per op offset into the debug info indices
  diIndicesNum: varint      // Total number of debug info indices
  padding: bytes            // Align to 4 bytes
  diIndices: uint64_t[]     // Array of debug indices to debug info attributes
  diAttrNum: varint         // Total number of debug info attributes
  padding: bytes            // Align to 4 bytes
  diOffsets: uint32_t[]     // Per debug info attribute offset into the debug info data
  diData: bytes             // Data for each debug info attribute
}
```

The debug section uses a multi-level indirection scheme:

- **Operations:** diOpsNum operations have debug info, with diIndexOffsets pointing into the indices array
- **Indices:** diIndicesNum total indices in diIndices, referencing debug info attributes
- **Attributes:** diAttrNum debug info attributes stored in diData with offsets in diOffsets

**Debug Section Multi-Level Indirection Diagram:**

```
+-----------------------------------------------------------------+
| Debug Section                                                    |
+-----------------------------------------------------------------+
| Level 1: Operation -> Index mapping                              |
|   diOpsNum: varint                                              |
|   diIndexOffsets: uint32_t[]  -> points into diIndices           |
+-----------------------------------------------------------------+
| Level 2: Index -> Attribute mapping                              |
|   diIndicesNum: varint                                          |
|   diIndices: uint64_t[]       -> points into diData              |
+-----------------------------------------------------------------+
| Level 3: Attribute Data                                          |
|   diAttrNum: varint                                            |
|   diOffsets: uint32_t[]       -> offsets within diData           |
|   diData: bytes               -> actual debug info entries       |
+-----------------------------------------------------------------+
```

Each debug info attribute begins with a debugEntryType indicating what kind of debug info it is:

| Value | Meaning |
|-------|---------|
| 0x00 | Unknown |
| 0x01 | DICompileUnit |
| 0x02 | DIFile |
| 0x03 | DILexicalBlock |
| 0x04 | DILoc |
| 0x05 | DISubprogram |
| 0x06 | CallSite |

debugEntryPayload describes line, file, variable name, function index, instruction offset, etc. Each debugEntryType has a fixed, known structure. If functionLocIndex[i] or an instruction's locationIndex is non-zero, it references debugEntryOffset[...] in this section, whose payload can store file/line info or other metadata.

**Debug Entry Payload Formats**

Each debug entry in diData begins with a debugEntryType byte followed by type-specific data:

```
debugEntry {
  debugEntryType : byte     // Identifies the debug info type
  entryData : byte[]        // Type-specific payload (format below)
}
```

The following debug entry types are supported:

Unknown Debug Info (debugEntryType = 0x00)

```c
debugEntryType : varint = 0x00  // Unknown debug info
// No additional payload
```

DICompileUnit (debugEntryType = 0x01)

```c
debugEntryType : varint = 0x01  // DICompileUnit
language : varint               // Source language identifier
fileIndex : varint              // Index of associated DIFile
producer : varint               // String index for compiler producer
optimized : byte                // 0x00=false, 0x01=true
emissionKind : varint           // Emission kind enumeration
```

DIFile (debugEntryType = 0x02)

```c
debugEntryType : varint = 0x02  // DIFile
filename : varint               // String index for filename
directory : varint              // String index for directory
```

DILexicalBlock (debugEntryType = 0x03)

```c
debugEntryType : varint = 0x03  // DILexicalBlock
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of parent scope (DIFile or DISubprogram)
```

DILoc (debugEntryType = 0x04)

```c
debugEntryType : varint = 0x04  // DILoc (source location)
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of scope (DISubprogram, DILexicalBlock, etc.)
inlinedAtIndex : varint         // Index of inlined location (0 if not inlined)
```

DISubprogram (debugEntryType = 0x05)

```c
debugEntryType : varint = 0x05  // DISubprogram (function debug info)
name : varint                   // String index for function name
linkageName : varint            // String index for linkage name
fileIndex : varint              // Index of associated DIFile
line : varint                   // Line number where function is defined
typeIndex : varint              // Index of function type
scopeLineIndex : varint         // Line number where scope begins
flags : varint                  // Function flags (visibility, etc.)
unitIndex : varint              // Index of associated DICompileUnit
```

CallSite (debugEntryType = 0x06)

```c
debugEntryType : varint = 0x06  // CallSite location
calleeIndex : varint            // Index of called location
callerIndex : varint            // Index of calling location
```

## Operation Opcodes and Encodings

Each instruction in Tile IR bytecode is represented as:

```
opcode : byte
locationIndex : varint
instructionSpecificFields : byte[]
```

In this representation, opcode uniquely identifies the operation. This matches one of the operations defined in the Tile IR dialect. locationIndex is always present; if the instruction does not carry debug info, this field is 0. A non-zero value refers to an entry in the Debug Section that contains file, line, or other source-level metadata. The instruction-specific fields (operands, attributes, etc.) follow a layout defined by the opcode.

Additionally, we do not store a resultIndex for each producing instruction. Instead, we rely on a sequential pass to assign local value indices at parse-time.

Older parsers are expected to skip or reject unknown opcodes. Over time, new operations can be added simply by assigning new opcodes and defining their binary payload formats.

### Operation Encoding Details

The general structure for operation encoding follows a consistent pattern, but varies based on the operation's characteristics:

**General Operation Structure**

```
opcode : byte                     // Operation identifier
locationIndex : varint            // Debug location (0 = no debug info)
resultTypes : typeIndex[]?        // Present for variadic result operations
flags : varint?                   // Optional flags for operations with optional fields
attributes : encoded_attr[]       // Operation-specific attributes
operands : operand_encoding       // Operation-specific operand encoding
regions : region_encoding[]?      // Present for operations with regions
```

**Flags Field Encoding**

For operations with optional attributes or operands, a flags field is used:

```
flags : varint                    // Bitfield encoding optional presence
```

The flags field uses individual bits to indicate the presence of optional attributes and operands:

- Bits 0-N: Optional attributes (in declaration order)
- Bits N+1-M: Optional operands (in declaration order)

UnitAttr attributes are encoded only in the flags field -- no additional data is written.

```
flags byte bit layout:
+---+---+---+---+---+---+---+---+
| ... | OptOp2 | OptOp1 | OptAttr2 | OptAttr1 |
+---+---+---+---+---+---+---+---+

Bit 0: First optional attribute present
Bit 1: Second optional attribute present
...
Bit N: Nth optional attribute present
Bit N+1: First optional operand group present
Bit N+2: Second optional operand group present
...
```

**Operand Encoding Patterns**

Operands are encoded differently based on the operation's operand structure:

- **Fixed Operands:** Written as sequential operand indices
- **Variadic Operands:** Prefixed with operand count, then indices
- **AttrSizedOperandSegments:** Each operand group encoded separately

```c
// Fixed operands (e.g., binary operations)
operand1Index : varint
operand2Index : varint

// Variadic operands (e.g., function calls)
numOperands : varint
operandIndices : varint[numOperands]

// AttrSizedOperandSegments (e.g., operations with optional operand groups)
group1 : optional_operand_group
group2 : optional_operand_group
...
```

**Result Type Encoding**

- **Fixed Results:** No result types encoded (inferred from operation)
- **Variadic Results:** Number of results encoded, followed by type indices

```c
// Variadic results
numResults : varint
resultTypeIndices : varint[numResults]
```

### Region Encoding

Operations with regions encode them after operands:

```
numRegions : varint
regions : region[numRegions]

region {
  numBlocks : varint
  blocks : block[numBlocks]
}

block {
  numArgs : varint
  argTypeIndices : varint[numArgs]
  numOps : varint
  operations : operation[numOps]
}
```

**Region Encoding Diagram:**

```
+---------------------------------------------------+
| Region                                             |
+---------------------------------------------------+
| numBlocks: varint (e.g., 2)                       |
+---------------------------------------------------+
| Block 0 (entry block):                             |
|   numArgs: varint (e.g., 2)                       |
|   argTypeIndices[0]: varint -> I32                |
|   argTypeIndices[1]: varint -> F32                |
|   numOps: varint (e.g., 3)                        |
|   operation[0]: ...                               |
|   operation[1]: ...                               |
|   operation[2]: ...                               |
+---------------------------------------------------+
| Block 1:                                           |
|   numArgs: varint (e.g., 1)                       |
|   argTypeIndices[0]: varint -> Token              |
|   numOps: varint (e.g., 2)                        |
|   operation[0]: ...                               |
|   operation[1]: ...                               |
+---------------------------------------------------+
```

### Common Operation Examples

#### Arithmetic Operations (e.g., cuda_tile.add)

```c
opcode : byte                     // e.g., 0x15 for add
locationIndex : varint            // Debug location
lhs : varint                      // Left operand index
rhs : varint                      // Right operand index
// Result type inferred from operands
```

**Example: Encoding `c = add(a, b)` with no debug info**

```
Hex bytes: 0x15 0x00 0x00 0x01
// opcode=0x15 (add), locationIndex=0 (no debug), lhs=0 (operand 0), rhs=1 (operand 1)
// Result is implicitly assigned value index 2
```

#### Memory Operations (e.g., cuda_tile.load)

```c
opcode : byte                     // e.g., 0x20 for load
locationIndex : varint            // Debug location
resultType : varint               // Type index for loaded value
address : varint                  // Address operand index
// Optional attributes encoded via flags
```

**Example: Encoding a load from an address with cache hints**

```
Hex bytes: 0x20 0x00 0x07 0x02 0x01 ...
// opcode=0x20 (load), locationIndex=0, resultType=7 (F32 Tile),
// address=2 (operand 2), flags=0x01 (has optional cache hint)
```

#### Control Flow Operations (e.g., cuda_tile.if)

```c
opcode : byte                     // e.g., 0x30 for if
locationIndex : varint            // Debug location
condition : varint                // Condition operand index
numRegions : varint               // Always 2 for if (then, else)
thenRegion : region               // Then block
elseRegion : region               // Else block (may be empty)
```

**Example: Encoding an if-then-else**

```
Hex bytes:
  0x30              // opcode (if)
  0x00              // locationIndex (no debug)
  0x05              // condition = operand 5
  0x02              // numRegions = 2 (then + else)
  // thenRegion:
    0x01            // numBlocks = 1
    0x00            // numArgs = 0
    0x03            // numOps = 3
    ...             // operations
  // elseRegion:
    0x01            // numBlocks = 1
    0x00            // numArgs = 0
    0x00            // numOps = 0 (empty else)
```

#### Function Call Operation (e.g., cuda_tile.call)

```c
opcode : byte                     // Call opcode
locationIndex : varint            // Debug location
calleeIndex : varint              // Index into function table
numArgs : varint                  // Number of arguments
argIndices : varint[numArgs]      // Argument operand indices
// Result types encoded for variadic results
numResults : varint               // Number of results
resultTypeIndices : varint[numResults] // Result type indices
```

#### Loop Operation (e.g., cuda_tile.for)

```c
opcode : byte                     // For loop opcode
locationIndex : varint            // Debug location
lowerBound : varint               // Lower bound operand index
upperBound : varint               // Upper bound operand index
step : varint                     // Step operand index
// Optional initArgs (token-ordered)
flags : varint                    // Flags for optional fields
numInitArgs : varint?             // Present if init args flag set
initArgIndices : varint[]?        // Present if init args flag set
// Loop body region
numRegions : varint               // Always 1
bodyRegion : region               // Loop body
// Result types
numResults : varint               // Number of result values
resultTypeIndices : varint[]      // Result type indices
```

## Section Ordering

Tile IR bytecode readers can handle sections in any order due to their flexible parsing design. The reader first discovers all sections and stores their payloads, then processes them in dependency order.

**Default Writing Order**

Writers typically emit sections in the following order:

1. Header (magic number + version)
2. Global Section (Section ID: 0x06) - Optional, only if globals present
3. Function Table Section (Section ID: 0x02) - Required
4. Constant Data Section (Section ID: 0x04) - Optional, only if constants present
5. Debug Section (Section ID: 0x03) - Optional
6. Type Section (Section ID: 0x05) - Required
7. String Section (Section ID: 0x01) - Required
8. End-of-Bytecode Marker (0x00) - Required

However, readers are not dependent on this order and can process sections regardless of their arrangement in the file. This flexibility enables future optimizations and different writing strategies.

**Section ID Summary:**

| Section ID | Section Name | Required | Description |
|-----------|-------------|----------|-------------|
| 0x00 | End-of-Bytecode | Yes | Marks end of bytecode stream |
| 0x01 | String Section | Yes | All textual names and strings |
| 0x02 | Function Table | Yes | Functions and their instruction bodies |
| 0x03 | Debug Section | No | Debug information |
| 0x04 | Constant Data | No | Large constants (tensors, arrays) |
| 0x05 | Type Section | Yes | Type definitions |
| 0x06 | Global Section | No | Module-level global variables |

**Reader Implementation**

The reader implements this flexibility by:

- **Discovery Phase:** Reads all section headers and stores their payloads in memory
- **Processing Phase:** Processes sections in dependency order regardless of file order
- **Lazy Resolution:** Resolves forward references (e.g., types, strings) on-demand

This design allows for efficient random access to any section and supports future file format optimizations.

**Section Dependency Graph**

```
                    +-----------------+
                    |  String Section |
                    |    (0x01)       |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
     +--------+----+ +------+------+ +-----+-------+
     | Type Section| |Debug Section| |Global Section|
     |   (0x05)    | |   (0x03)    | |    (0x06)    |
     +------+------+ +------+------+ +------+-------+
             |              |              |
             v              |              v
     +-------+--------+     |     +-------+--------+
     |Constant Section|     |     |Constant Section|
     |    (0x04)      |     |     |    (0x04)      |
     +-------+--------+     |     +-------+--------+
             |              |              |
             +-------+------+--------------+
                     |
                     v
            +--------+--------+
            |Function Table   |
            |    (0x02)       |
            +-----------------+

Arrows indicate "references" or "depends on":
  - Function table references: types, strings, constants, debug info
  - Global section references: types, strings, constants
  - Debug section references: strings
  - Type section references: strings (for named types)
  - All sections may reference the string section
```

**Processing Order (Dependency Resolution):**

```
1. Load String Section    (no dependencies)
2. Load Type Section      (depends on: strings)
3. Load Constant Section  (depends on: strings)
4. Load Debug Section     (depends on: strings)
5. Load Global Section    (depends on: strings, types, constants)
6. Load Function Table    (depends on: strings, types, constants, debug)
```
