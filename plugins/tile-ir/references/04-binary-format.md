# 4. Binary Format and Bytecode

This document provides comprehensive coverage of the Tile IR binary format -- the stable, versioned bytecode representation used for serialization, storage, and transport of Tile IR modules. The binary format is designed for efficient loading, forward and backward compatibility, and human inspectability.

**Tile IR Specification Version:** 13.2 (March 2026)

---

## Table of Contents

1. [Overview](#41-overview)
2. [Primitives](#42-primitives)
3. [File Structure](#43-file-structure)
4. [Type Encodings](#44-type-encodings)
5. [Attribute Encodings](#45-attribute-encodings)
6. [Operation Encoding](#46-operation-encoding)
7. [Section Ordering](#47-section-ordering)
8. [Encoding Examples](#48-encoding-examples)

---

## 4.1 Overview

Tile IR uses a **stable, versioned bytecode format** for representing compiled programs. The format is the canonical interchange mechanism between the Tile IR compiler frontends, optimizing middle-end, and the GPU driver runtime. Every conforming Tile IR implementation must be able to parse and execute valid bytecode files.

### 4.1.1 Design Goals

The binary format is governed by the following design goals, listed in priority order:

1. **Stability** -- The bytecode format is versioned and guarantees that bytecode produced by a conforming serializer can be loaded by all conforming deserializers of the same or later version. The format never silently changes semantics.

2. **Expandable operations** -- New operations, types, and attributes can be added in future versions without breaking existing parsers. Unknown opcodes are handled gracefully through well-defined skip mechanisms.

3. **Minimal type system** -- The binary format encodes only the types necessary for correct execution. Derived or sugar types are desugared before serialization, keeping the on-disk representation lean.

4. **Human inspectable** -- The format uses recognizable magic bytes, readable section identifiers, and straightforward encodings so that standard binary inspection tools (e.g., `xxd`, `hexdump`) can reveal high-level structure without a dedicated disassembler.

5. **Lazy loading** -- Sections are independently decodable. A consumer can read the string table and function table without parsing all operations, enabling fast startup and partial loading for tooling.

6. **Compact representation** -- Variable-width integer encoding (VarInts) keeps the format small for typical programs where most integer values are small. Repeated strings are deduplicated through the string section.

### 4.1.2 Compatibility Guarantees

The Tile IR binary format provides both forward and backward compatibility:

- **Backward compatibility** -- A new deserializer can always read bytecode produced by an older serializer. New fields added in later versions are simply absent in older files, and the deserializer fills in default values.

- **Forward compatibility** -- An old deserializer encountering bytecode from a newer version handles it gracefully. Sections and fields it does not understand are skipped using their declared lengths. Operations with unknown opcodes are rejected with a clear error message rather than causing undefined behavior.

- **Version targeting** -- A serializer may target a specific older version of the format, omitting features introduced in later versions. This enables cross-version toolchains where the deployment environment runs an older driver.

These guarantees are enforced through the version header and the self-describing section structure described in subsequent sections.

### 4.1.3 Notation Conventions

Throughout this chapter, the following notation conventions are used:

| Notation | Meaning |
|----------|---------|
| `byte` | An unsigned 8-bit integer (0x00 to 0xFF) |
| `uint8` | An unsigned 8-bit integer |
| `uint16` | A little-endian unsigned 16-bit integer |
| `uint32` | A little-endian unsigned 32-bit integer |
| `uint64` | A little-endian unsigned 64-bit integer |
| `varint` | A variable-width unsigned integer (see Section 4.2.2) |
| `section[]` | A sequence of zero or more sections |
| `field?` | An optional field |
| `0xNN` | A hexadecimal literal |
| `"str"` | An ASCII string literal |
| `[N]` | An array of N elements |
| `{...}` | A grouped structure |

Binary diagrams use a horizontal layout where increasing byte offsets go from left to right and subsequent lines represent increasing offsets:

```
+--------+--------+--------+--------+
| byte 0 | byte 1 | byte 2 | byte 3 |
+--------+--------+--------+--------+
| byte 4 | byte 5 | byte 6 | byte 7 |
+--------+--------+--------+--------+
```

---

## 4.2 Primitives

The Tile IR binary format is built from a small set of primitive encodings. All multi-byte values use little-endian byte order unless explicitly stated otherwise.

### 4.2.1 Fixed-Width Integers

Fixed-width integers are encoded in **little-endian byte order** (least significant byte first). This matches the native byte order of all supported GPU host platforms (x86-64 and ARM64).

```
byte ::= 0x00 ... 0xFF              (8 bits, unsigned)

uint16 ::= byte[2]                   (16 bits, little-endian)
  encoding: low_byte || high_byte
  value:    low_byte + high_byte * 256

uint32 ::= byte[4]                   (32 bits, little-endian)
  encoding: b0 || b1 || b2 || b3
  value:    b0 + b1*256 + b2*65536 + b3*16777216

uint64 ::= byte[8]                   (64 bits, little-endian)
  encoding: b0 || b1 || ... || b7
  value:    sum(b_i * 256^i for i in 0..7)
```

**Example:** The value `0x12345678` is encoded as four bytes:

```
Offset:  0x00  0x01  0x02  0x03
Bytes:   0x78  0x56  0x34  0x12
```

Signed integers use two's complement representation but are generally avoided in the binary format in favor of unsigned encodings with explicit sign handling at the semantic level.

### 4.2.2 Variable-Width Integers (VarInts)

Variable-width integers (VarInts) are used throughout the format for values that are frequently small but may need to represent the full 64-bit range. Tile IR uses the **PrefixVarInt** encoding, a variant of LEB128 optimized for fast decoding.

#### Encoding Scheme

A PrefixVarInt uses 1 to 9 bytes to encode a 64-bit unsigned value. The first byte determines the total length through a prefix pattern in its most significant bits:

| First byte prefix | Total bytes | Value bits per byte | Maximum value |
|-------------------|-------------|---------------------|---------------|
| `0xxxxxxx` | 1 | 7 | 127 |
| `10xxxxxx` | 2 | 7 | 16,383 |
| `110xxxxx` | 3 | 7 | 2,097,151 |
| `1110xxxx` | 4 | 7 | 268,435,455 |
| `11110xxx` | 5 | 7 | 34,359,738,367 |
| `111110xx` | 6 | 7 | 4,398,046,511,103 |
| `1111110x` | 7 | 7 | 562,949,953,421,311 |
| `11111110` | 8 | 7 | 72,057,594,037,927,935 |
| `11111111` | 9 | 8 | 18,446,744,073,709,551,615 |

#### Decoding Algorithm

```
function decode_prefix_varint(bytes):
    first = bytes[0]

    if first < 0x80:                          # 0xxxxxxx
        return first                           # 1 byte, 7 bits

    if first < 0xC0:                           # 10xxxxxx
        return ((first & 0x3F) << 8) | bytes[1]  # 2 bytes, 14 bits

    if first < 0xE0:                           # 110xxxxx
        return ((first & 0x1F) << 16) | (bytes[1] << 8) | bytes[2]

    if first < 0xF0:                           # 1110xxxx
        return ((first & 0x0F) << 24) | (bytes[1] << 16)
             | (bytes[2] << 8) | bytes[3]

    if first < 0xF8:                           # 11110xxx
        return ((first & 0x07) << 32) | (bytes[1] << 24)
             | (bytes[2] << 16) | (bytes[3] << 8) | bytes[4]

    if first < 0xFC:                           # 111110xx
        return ((first & 0x03) << 40) | (bytes[1] << 32)
             | (bytes[2] << 24) | (bytes[3] << 16)
             | (bytes[4] << 8) | bytes[5]

    if first < 0xFE:                           # 1111110x
        return ((first & 0x01) << 48) | (bytes[1] << 40)
             | (bytes[2] << 32) | (bytes[3] << 24)
             | (bytes[4] << 16) | (bytes[5] << 8)
             | bytes[6]

    if first == 0xFE:                          # 11111110
        return (bytes[1] << 48) | (bytes[2] << 40)
             | (bytes[3] << 32) | (bytes[4] << 24)
             | (bytes[5] << 16) | (bytes[6] << 8)
             | bytes[7]

    # first == 0xFF                             # 11111111
    return (bytes[1] << 56) | (bytes[2] << 48)
         | (bytes[3] << 40) | (bytes[4] << 32)
         | (bytes[5] << 24) | (bytes[6] << 16)
         | (bytes[7] << 8) | bytes[8]
```

**Note on the 9-byte case:** When the first byte is `0xFF`, the subsequent 8 bytes are interpreted as a raw little-endian `uint64`, providing the full 64-bit range. In this case, each of the 8 following bytes contributes all 8 bits (not 7), giving 8 * 8 = 64 bits of value.

#### Encoding Examples

| Value | Hex bytes | Length |
|-------|-----------|--------|
| 0 | `0x00` | 1 |
| 1 | `0x01` | 1 |
| 127 | `0x7F` | 1 |
| 128 | `0x80 0x80` | 2 |
| 255 | `0x81 0xFF` | 2 |
| 300 | `0x82 0x2C` | 2 |
| 16383 | `0xBF 0xFF` | 2 |
| 16384 | `0xC0 0x40 0x00` | 3 |
| 65535 | `0xC0 0xFF 0xFF` | 3 |

#### Rationale

PrefixVarInt is preferred over standard LEB128 for two reasons:

1. **Single-pass decoding** -- The length is determined from the first byte alone, allowing immediate allocation and reading. Standard LEB128 requires examining the high bit of every byte sequentially.
2. **Better branch prediction** -- The length prefix pattern produces predictable branches in the decoder, improving performance on modern CPUs.

### 4.2.3 Byte Sequences

A byte sequence is encoded as a length prefix followed by the raw bytes:

```
byte_sequence ::= length:varint, byte[length]
```

The length field specifies the number of bytes that follow. A length of zero indicates an empty sequence.

### 4.2.4 String Encoding

Strings are stored in the **String Section** (Section 4.3.3) and referenced by index throughout the bytecode. Individual string entries are encoded as:

```
string_entry ::= length:varint, utf8_byte[length]
```

Strings are UTF-8 encoded and are **not** null-terminated in the binary format. The length prefix includes only the UTF-8 content bytes, not a trailing null.

---

## 4.3 File Structure

A Tile IR bytecode file consists of a fixed magic number, a version header, and a sequence of sections.

### 4.3.0 Top-Level Structure

```
bytecode {
    magic:    "\x7FTileIR\x00"                    // 8 bytes, fixed
    version:  { uint8 major, uint8 minor, uint16 tag }  // 4 bytes, fixed
    sections: section[]                             // variable length
}
```

The total file size is determined by reading sections until end-of-file. There is no explicit file-length field; instead, each section carries its own length, and the reader processes sections until the stream is exhausted.

### 4.3.1 Magic Number

The first 8 bytes of every Tile IR bytecode file are the fixed magic number:

```
Offset:  0x00  0x01  0x02  0x03  0x04  0x05  0x06  0x07
Bytes:   0x7F  0x54  0x69  0x6C  0x65  0x49  0x52  0x00
Chars:   DEL   T     i     l     e     I     R     NUL
```

This is the bytes `0x7F` followed by the ASCII string `TileIR` followed by a null byte `0x00`.

The leading `0x7F` byte ensures the file is not confused with a text file (which would start with a printable ASCII character) and follows a convention similar to ELF (`\x7FELF`).

The magic number serves three purposes:

1. **Identification** -- Allows `file(1)` and similar tools to identify Tile IR bytecode.
2. **Validation** -- A quick sanity check before proceeding with parsing.
3. **Safety** -- The `0x7F` byte prevents the file from being interpreted as a shell script or text file even if accidentally executed.

### 4.3.2 Version

Immediately after the magic number is a 4-byte version structure:

```
+--------+--------+--------+--------+
| major  | minor  |    tag (LE)      |
| uint8  | uint8  |     uint16       |
+--------+--------+--------+--------+
Offset:  0x08     0x09     0x0A     0x0B
```

The three fields are:

| Field | Type | Description |
|-------|------|-------------|
| `major` | `uint8` | Major version number. Incremented for breaking format changes. |
| `minor` | `uint8` | Minor version number. Incremented for backward-compatible additions. |
| `tag` | `uint16` | Pre-release or vendor tag. Zero for stable releases. Non-zero values indicate development, experimental, or vendor-specific variants. |

**Current version:** Major = 13, Minor = 2, Tag = 0 (encoding: `0x0D 0x02 0x00 0x00`)

#### Version Comparison Rules

A deserializer compares its own supported version against the file version as follows:

1. If the file's **major** version is greater than the deserializer's maximum supported major version, the file must be **rejected**. The format has changed in a way the deserializer cannot handle.

2. If the file's major version equals the deserializer's major version, and the file's minor version is less than or equal to the deserializer's maximum supported minor version, the file is **accepted**.

3. If the file's major version equals the deserializer's major version, and the file's minor version is **greater** than the deserializer's maximum supported minor version, the deserializer should attempt to read the file. Any sections, types, operations, or attributes it does not recognize should be handled according to the forward compatibility rules (skip unknown sections, reject unknown opcodes with a descriptive error).

4. If the file's **tag** is non-zero, the deserializer may emit a warning but should still attempt to parse the file. Vendor-specific tags may indicate extensions that the standard deserializer does not understand.

#### Version Targeting

A serializer can produce bytecode targeting an older version by:

- Omitting operations, types, or attributes introduced in newer versions.
- Using the older version's encoding rules for any features that changed encoding.
- Setting the version header to the target version, not the serializer's own version.

This is useful when the deployment environment is known to run an older driver version.

### 4.3.3 Sections

Following the version header, the remainder of the file consists of a sequence of sections. Each section is self-describing with its own length, allowing unknown sections to be skipped.

#### Section Structure

```
section {
    idAndIsAligned: uint8          // Section ID and alignment flag
    length:         varint         // Byte length of section data (excluding padding)
    alignment?:     varint         // Present if alignment bit is set; alignment requirement
    padding:        byte[]         // 0xCB padding bytes to reach alignment
    data:           byte[length]   // Section payload
}
```

#### Section Header: idAndIsAligned

The first byte of each section header encodes both the section identifier and whether the section has a custom alignment requirement:

```
+--------+--------+--------+--------+--------+--------+--------+--------+
| bit 7  | bit 6  | bit 5  | bit 4  | bit 3  | bit 2  | bit 1  | bit 0  |
+--------+--------+--------+--------+--------+--------+--------+--------+
| align  |                  section_id (7 bits)                            |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

- **Bit 7 (align)** -- If set (1), the section has a custom alignment requirement. The `alignment` field follows the `length` field. If clear (0), the section data immediately follows the `length` field with no alignment constraint beyond natural byte alignment.

- **Bits 6-0 (section_id)** -- The section identifier, a 7-bit unsigned integer (0x00 to 0x7F).

#### Section Length

The `length` field is a varint specifying the number of bytes in the `data` payload. It does **not** include the bytes consumed by the section header (`idAndIsAligned`, `length`, `alignment`, or `padding`).

#### Section Alignment

When the alignment bit is set, the `alignment` field (a varint) specifies the byte alignment requirement for the start of the section data. The alignment value must be a power of 2.

After reading the alignment value, the serializer must insert `0xCB` padding bytes until the current file offset is a multiple of the alignment value. The value `0xCB` is chosen as a distinctive padding byte that is unlikely to appear as valid data, aiding in debugging and forensic analysis.

A reader computing the padding required after the alignment field:

```
current_offset = position_after_alignment_field
target_offset = ceil(current_offset / alignment) * alignment
padding_bytes = target_offset - current_offset
verify: all padding bytes equal 0xCB
data_start = target_offset
data_end = data_start + length
```

#### Section IDs

The following section IDs are defined:

| ID | Name | Required | Description |
|----|------|----------|-------------|
| 0x00 | _Reserved_ | -- | Reserved for future use; must not appear in bytecode |
| 0x01 | String Section | Yes | String table for all string references |
| 0x02 | Function Table Section | Yes | Describes kernel entry points |
| 0x03 | Debug Section | Optional | Debug information (locations, scopes) |
| 0x04 | Constant Data Section | Optional | Inline constant data blocks |
| 0x05 | Type Section | Yes | Type definitions referenced by operations |
| 0x06 | Global Section | Optional | Global variable definitions |
| 0x07-0x7F | _Unassigned_ | -- | Reserved for future section types |

---

### 4.3.4 String Section (ID 0x01)

The String Section contains a table of UTF-8 strings referenced by index throughout the bytecode. All string values -- operation names, attribute keys, global variable names, kernel names, debug file paths -- are stored here and referenced by their 0-based index.

```
string_section {
    count: varint          // Number of string entries
    entries: string[count] // String entries
}

string {
    length: varint         // Number of UTF-8 bytes
    data: byte[length]     // UTF-8 encoded string content
}
```

**String reference:** When another section refers to a string, it uses a varint index into this table. Index 0 refers to the first string, index 1 to the second, and so on.

**Deduplication:** Serializers should deduplicate identical strings to minimize file size. Deserializers treat the index as a position-based lookup and do not require deduplication.

**Empty string:** The empty string (length = 0) is a valid entry.

**Example encoding of a string table with three entries:**

```
Strings: ["hello", "world", "cuda_tile.module"]

Encoding:
  count: 0x03                                 // 3 entries
  entry[0]: 0x05 'h' 'e' 'l' 'l' 'o'         // "hello", 5 bytes
  entry[1]: 0x05 'w' 'o' 'r' 'l' 'd'         // "world", 5 bytes
  entry[2]: 0x10 'c' 'u' 'd' 'a' '_' 't' 'i' 'l' 'e' '.' 'm' 'o' 'd' 'u' 'l' 'e'
                                               // "cuda_tile.module", 16 bytes
```

---

### 4.3.5 Function Table Section (ID 0x02)

The Function Table Section lists all kernel entry points in the module. Each entry describes the kernel's name, signature, and location within the operation stream.

```
function_table_section {
    count: varint                // Number of function entries
    entries: function_entry[count]
}

function_entry {
    name_index: varint           // String table index of kernel name
    entry_flag: byte             // Flags (see below)
    num_params: varint           // Number of parameters
    param_types: type_ref[num_params]  // Type references for parameters
    body_offset: varint          // Byte offset of kernel body in the Type Section's operation stream
    body_length: varint          // Byte length of the kernel body
}
```

#### Entry Flag Encoding

The `entry_flag` byte encodes kernel properties:

```
+--------+--------+--------+--------+--------+--------+--------+--------+
| bit 7  | bit 6  | bit 5  | bit 4  | bit 3  | bit 2  | bit 1  | bit 0  |
+--------+--------+--------+--------+--------+--------+--------+--------+
|  0-0   |  0-0   |  0-0   |  0-0   |  0-0   |  0-0   |extern  | kernel |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `kernel` | Set if this is a kernel entry point (callable from host) |
| 1 | `extern` | Set if this function is externally visible (callable from other modules) |
| 2-7 | _Reserved_ | Must be zero; reserved for future use |

**Notes:**

- A `kernel` entry point is the standard form -- it represents a Tile IR kernel launched from the host via the CUDA driver API.
- An `extern` function is visible for inter-module linking but is not directly launchable as a kernel.
- If both bits are clear (0x00), the function is internal to the module and not accessible externally.
- The `type_ref` for each parameter is an index into the Type Section's type table (see Section 4.4).

---

### 4.3.6 Constant Data Section (ID 0x04)

The Constant Data Section stores inline binary data for constant values that cannot be represented as simple scalar literals.

```
constant_data_section {
    count: varint                     // Number of constant data blocks
    entries: constant_block[count]
}

constant_block {
    alignment: varint                 // Alignment requirement (power of 2)
    data_length: varint               // Byte length of data
    padding: byte[]                   // 0xCB padding to alignment
    data: byte[data_length]           // Raw binary data
}
```

Constant data blocks are referenced by other sections (e.g., DenseElements attributes) through their 0-based index. The alignment field ensures that vectorized load instructions can correctly read the constant data at runtime without misalignment penalties.

---

### 4.3.7 Type Section (ID 0x05)

The Type Section contains all type definitions used in the module, along with the operation stream for function bodies.

```
type_section {
    type_count: varint               // Number of type definitions
    types: type_def[type_count]      // Type definitions
    op_stream_length: varint         // Byte length of the combined operation stream
    op_stream: byte[op_stream_length] // All function bodies concatenated
}
```

Each type definition encodes a complete Tile IR type. The operation stream contains the encoded operations for all function bodies; the Function Table Section's `body_offset` and `body_length` fields reference into this stream.

---

### 4.3.8 Global Section (ID 0x06)

The Global Section defines module-level global variables.

```
global_section {
    count: varint                    // Number of global variables
    entries: global_entry[count]
}

global_entry {
    name_index: varint               // String table index of global name
    type_ref: varint                 // Type section index of global type
    is_mutable: byte                 // 0 = constant, 1 = mutable
    initializer?: constant_initializer  // Present if is_mutable == 0
}

constant_initializer {
    tag: byte                        // Initializer kind
    data: ...                        // Depends on tag
}
```

Initializer tags:

| Tag | Name | Description |
|-----|------|-------------|
| 0x00 | Uninitialized | No initial value; contents undefined |
| 0x01 | Zero | Zero-initialized (all bits zero) |
| 0x02 | Scalar | A single scalar value, encoded per element type |
| 0x03 | Dense | Dense tensor data from Constant Data Section |
| 0x04 | External | Symbol reference (linker resolved) |

---

### 4.3.9 Debug Section (ID 0x03)

The Debug Section contains optional debug information including source locations, file references, scope metadata, and compilation unit information.

```
debug_section {
    version: varint                  // Debug info format version
    compile_unit_count: varint       // Number of compilation units
    compile_units: compile_unit[compile_unit_count]
    file_table: file_table           // Source file references
    location_table: location_table   // Source location mappings
}
```

See Chapter 9 (Debug Information) for the complete debug section format.

---

## 4.4 Type Encodings

Each type in the Type Section is encoded as a tagged union. The first byte (the **type tag**) identifies the kind of type, followed by any type-specific payload.

### 4.4.1 Type Tag Table

The complete set of type tags is:

| Tag | Name | Payload | Description |
|-----|------|---------|-------------|
| 0x00 | I1 | none | 1-bit integer (predicate) |
| 0x01 | I8 | none | 8-bit signless integer |
| 0x02 | I16 | none | 16-bit signless integer |
| 0x03 | I32 | none | 32-bit signless integer |
| 0x04 | I64 | none | 64-bit signless integer |
| 0x05 | F16 | none | IEEE 754 half-precision (binary16) |
| 0x06 | BF16 | none | Brain floating-point (bfloat16) |
| 0x07 | F32 | none | IEEE 754 single-precision (binary32) |
| 0x08 | TF32 | none | TensorFloat-32 (8 exp, 10 mantissa) |
| 0x09 | F64 | none | IEEE 754 double-precision (binary64) |
| 0x0A | F8E4M3FN | none | FP8 E4M3 FN (4 exp, 3 mantissa, no infinity, no NaN) |
| 0x0B | F8E5M2 | none | FP8 E5M2 (5 exp, 2 mantissa, with NaN/inf) |
| 0x0C | Pointer | element_type: type_ref | Typed pointer |
| 0x0D | Tile | shape, element_type | N-dimensional tile tensor |
| 0x0E | TensorView | shape, element_type, strides | Strided memory view |
| 0x0F | PartitionView | tile_shape, tensor_view: type_ref, dim_map | Tiled partition of a tensor view |
| 0x10 | Function | param_types, return_types | Function type signature |
| 0x11 | Token | none | Memory ordering token |
| 0x12-0xFF | _Reserved_ | -- | Reserved for future type extensions |

### 4.4.2 Element Type Encoding

Integer and floating-point element types are encoded as a single byte (the type tag) with no additional payload:

```
element_type ::= type_tag:byte    // One of 0x00 through 0x0B
```

### 4.4.3 Pointer Type Encoding

```
pointer_type {
    tag: 0x0C                      // Pointer tag
    element_type_ref: varint       // Index into type table for pointed-to type
}
```

**Example:** `ptr<f32>` is encoded as:

```
0x0C                    // Pointer tag
0x07                    // Type table index for F32 (assuming F32 is at index 7)
```

### 4.4.4 Tile Type Encoding

```
tile_type {
    tag: 0x0D                      // Tile tag
    rank: varint                   // Number of dimensions (0 for scalar)
    shape: varint[rank]            // Size of each dimension
    element_type_ref: varint       // Index into type table for element type
}
```

The rank may be zero, representing a scalar tile. All shape values must be positive integers that are powers of 2 (a Tile IR constraint enforced at validation time, not at the binary format level).

**Example:** `tile<128x64xf32>` is encoded as:

```
0x0D                    // Tile tag
0x02                    // Rank = 2
0x80 0x00               // Dim 0 = 128 (varint encoding)
0x40                    // Dim 1 = 64 (varint encoding)
0x07                    // Element type ref = F32
```

**Example:** `tile<i32>` (scalar tile) is encoded as:

```
0x0D                    // Tile tag
0x00                    // Rank = 0 (scalar)
0x03                    // Element type ref = I32
```

### 4.4.5 TensorView Type Encoding

```
tensor_view_type {
    tag: 0x0E                      // TensorView tag
    rank: varint                   // Number of dimensions
    shape: varint[rank]            // Size of each dimension (may use dynamic marker)
    element_type_ref: varint       // Index into type table for element type
    stride_count: varint           // Number of stride values (= rank)
    strides: varint[stride_count]  // Stride values (may use dynamic marker)
}
```

Dynamic dimensions and strides use the sentinel value `0xFFFFFFFFFFFFFFFF` (max uint64) to indicate a runtime-determined value. In the varint encoding, this sentinel requires 9 bytes.

**Example:** `tensor_view<?x?xf32, strides=[?,1]>` is encoded as:

```
0x0E                    // TensorView tag
0x02                    // Rank = 2
0xFF ... (9 bytes)      // Dim 0 = dynamic (max uint64)
0xFF ... (9 bytes)      // Dim 1 = dynamic (max uint64)
0x07                    // Element type ref = F32
0x02                    // Stride count = 2
0xFF ... (9 bytes)      // Stride 0 = dynamic
0x01                    // Stride 1 = 1
```

### 4.4.6 PartitionView Type Encoding

```
partition_view_type {
    tag: 0x0F                      // PartitionView tag
    tile_rank: varint              // Rank of the tile shape
    tile_shape: varint[tile_rank]  // Tile dimensions
    tensor_view_ref: varint        // Type table index for the underlying tensor view
    dim_map_count: varint          // Number of dimension mapping entries
    dim_map: varint[dim_map_count] // Dimension mapping values
}
```

The `dim_map` encodes how partition view dimensions map to tensor view dimensions. Each entry is an integer index into the tensor view's dimensions.

### 4.4.7 Function Type Encoding

```
function_type {
    tag: 0x10                      // Function tag
    num_params: varint             // Number of parameter types
    param_types: varint[num_params]   // Type table indices for parameters
    num_results: varint            // Number of result types
    result_types: varint[num_results] // Type table indices for results
}
```

### 4.4.8 Token Type Encoding

```
token_type {
    tag: 0x11                      // Token tag
    // No additional payload
}
```

---

## 4.5 Attribute Encodings

Attributes encode compile-time constant metadata attached to operations. Each attribute is encoded as a tagged value.

### 4.5.1 Attribute Tag Table

| Tag | Name | Payload | Description |
|-----|------|---------|-------------|
| 0x01 | Integer | value | Integer attribute value |
| 0x02 | Float | value | Floating-point attribute value |
| 0x03 | Bool | value | Boolean attribute value |
| 0x04 | Type | type_ref | Type attribute value |
| 0x05 | String | string_ref | String attribute value |
| 0x06 | Array | count, elements | Array of attributes |
| 0x07 | DenseElements | shape, data | Dense tensor of elements |
| 0x08 | DivBy | divisor | Divisibility constraint |
| 0x09 | SameElements | element | All elements are identical |
| 0x0A | Dictionary | count, entries | Key-value dictionary |
| 0x0B | OptimizationHints | flags | Optimization hint flags |
| 0x0C | NonNegative | none | Non-negative constraint annotation |

### 4.5.2 Integer Attribute (0x01)

```
integer_attribute {
    tag: 0x01
    bit_width: varint              // Bit width of the integer
    value: varint                  // Unsigned value (sign interpreted per context)
}
```

The `bit_width` field records the intended width (e.g., 32 for `i32`). The value is stored as an unsigned varint; sign interpretation is determined by the consuming operation.

### 4.5.3 Float Attribute (0x02)

```
float_attribute {
    tag: 0x02
    type_tag: byte                 // Element type tag (0x05-0x0B)
    value: byte[float_size]        // IEEE bit pattern in little-endian
}
```

The `float_size` is determined by the `type_tag`:

| type_tag | Type | float_size (bytes) |
|----------|------|--------------------|
| 0x05 | F16 | 2 |
| 0x06 | BF16 | 2 |
| 0x07 | F32 | 4 |
| 0x08 | TF32 | 4 |
| 0x09 | F64 | 8 |
| 0x0A | F8E4M3FN | 1 |
| 0x0B | F8E5M2 | 1 |

**Example:** The value `3.14` as `f32` (bit pattern `0x4048F5C3`) is encoded as:

```
0x02                    // Float attribute tag
0x07                    // F32 type tag
0xC3 0xF5 0x48 0x40    // IEEE 754 bit pattern, little-endian
```

### 4.5.4 Bool Attribute (0x03)

```
bool_attribute {
    tag: 0x03
    value: byte                    // 0x00 = false, 0x01 = true
}
```

### 4.5.5 Type Attribute (0x04)

```
type_attribute {
    tag: 0x04
    type_ref: varint               // Index into the type table
}
```

### 4.5.6 String Attribute (0x05)

```
string_attribute {
    tag: 0x05
    string_ref: varint             // Index into the string table
}
```

### 4.5.7 Array Attribute (0x06)

```
array_attribute {
    tag: 0x06
    count: varint                  // Number of elements
    elements: attribute[count]     // Each element is a full attribute encoding
}
```

Array attributes can be nested: an element may itself be an array attribute, a dictionary attribute, or any other attribute type.

### 4.5.8 DenseElements Attribute (0x07)

```
dense_elements_attribute {
    tag: 0x07
    element_type_ref: varint       // Type table index for element type
    rank: varint                   // Number of shape dimensions
    shape: varint[rank]            // Shape dimensions
    data_block_ref: varint         // Index into Constant Data Section
}
```

The DenseElements attribute represents a dense tensor of constant values. The raw data is stored in the Constant Data Section and referenced by index. The data is laid out in row-major order with the element type's natural byte width.

### 4.5.9 DivBy Attribute (0x08)

```
div_by_attribute {
    tag: 0x08
    divisor: varint                // The divisor value (must be > 0)
}
```

Indicates that the annotated value is known to be divisible by `divisor`. Used for alignment and stride constraints.

### 4.5.10 SameElements Attribute (0x09)

```
same_elements_attribute {
    tag: 0x09
    element: attribute             // The repeated element value
}
```

Represents a tensor where every element has the same value. Only the single element is stored.

### 4.5.11 Dictionary Attribute (0x0A)

```
dictionary_attribute {
    tag: 0x0A
    count: varint                  // Number of key-value pairs
    entries: dict_entry[count]
}

dict_entry {
    key_ref: varint                // String table index for key
    value: attribute               // Attribute value
}
```

Dictionary attributes are used for named properties on operations. Keys are string references; values can be any attribute type.

### 4.5.12 OptimizationHints Attribute (0x0B)

```
optimization_hints_attribute {
    tag: 0x0B
    flags: varint                  // Bitfield of optimization hints
}
```

Flag bits (defined to date):

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `noinline` | Do not inline this operation |
| 1 | `always_inline` | Always inline this operation |
| 2 | `fast_math` | Allow fast-math transformations |
| 3 | `unroll` | Suggest unrolling loops |
| 4 | `novectorize` | Do not vectorize |
| 5-63 | _Reserved_ | Reserved for future hints |

### 4.5.13 NonNegative Attribute (0x0C)

```
non_negative_attribute {
    tag: 0x0C
    // No additional payload
}
```

Annotates a value as being known non-negative at compile time. Used for optimization and validation of stride, shape, and index calculations.

---

## 4.6 Operation Encoding

Operations are encoded in a compact binary format within the operation stream of the Type Section. Each operation consists of a fixed header followed by variable-length sections for results, attributes, operands, and regions.

### 4.6.1 General Operation Structure

```
operation {
    opcode: varint                          // Operation kind
    location_index: varint                  // Debug location (string table or location table index)
    flags: byte                             // Operation-specific flags
    num_result_types: varint                // Number of result types (if has_results flag set)
    result_types: varint[num_result_types]  // Type table indices for results
    num_attributes: varint                  // Number of attributes
    attributes: attr_entry[num_attributes]  // Named attributes
    operand_encoding: ...                   // Depends on operand pattern (see 4.6.2)
    num_regions: varint                     // Number of nested regions
    regions: region[num_regions]            // Nested regions
}
```

#### Flags Byte

```
+--------+--------+--------+--------+--------+--------+--------+--------+
| bit 7  | bit 6  | bit 5  | bit 4  | bit 3  | bit 2  | bit 1  | bit 0  |
+--------+--------+--------+--------+--------+--------+--------+--------+
|  0-0   | region | attrs  | results|          reserved                    |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

| Bit | Name | Description |
|-----|------|-------------|
| 0-4 | reserved | Must be zero |
| 5 | `has_results` | If set, the operation produces results (num_result_types and result_types follow) |
| 6 | `has_attrs` | If set, the operation has attributes (num_attributes and attributes follow) |
| 7 | `has_regions` | If set, the operation contains regions (num_regions and regions follow) |

This flag-based encoding allows simple operations (e.g., `return` with no results, no attributes, no regions) to be encoded very compactly.

### 4.6.2 Operand Encoding Patterns

Operations declare their operands using one of three encoding patterns. The pattern is determined by the operation's definition (not encoded explicitly in the bytecode; the decoder knows the pattern from the opcode).

#### Fixed Operands

Fixed-operand operations have a known, constant number of operands determined by the opcode. The operands are simply encoded as a sequence of SSA value indices:

```
fixed_operands {
    operands: varint[N]            // N is fixed per opcode
}
```

Each `varint` is an index into the current function's SSA value table, referencing the value produced by a preceding operation (or a function parameter).

**Example:** `addf` takes exactly 2 operands (the two input tiles).

#### Variadic Operands

Variadic-operand operations have a variable number of operands, encoded with an explicit count:

```
variadic_operands {
    count: varint                  // Number of operands
    operands: varint[count]        // SSA value indices
}
```

**Example:** `cat` takes a variadic number of tile operands to concatenate.

#### AttrSizedOperandSegments

Some operations have multiple operand groups, where each group can be variadic. The sizes are encoded as an attribute of the operation. In the binary format, this is encoded as:

```
attr_sized_operands {
    total_count: varint            // Total number of operands across all groups
    operands: varint[total_count]  // All operands concatenated
    // Group sizes are stored as an attribute named "operand_segments"
}
```

The `operand_segments` attribute is an Array attribute of Integer attributes, one per group, specifying how many operands belong to each group. The decoder reads the total operand array and then splits it according to the segment sizes.

**Example:** The `for` operation has three operand groups:
1. Loop bounds (`lower`, `upper`, `step`) -- 3 operands
2. `iter_values` -- variadic
3. Body region arguments -- implicit from iter_values

### 4.6.3 Attribute Entry Encoding

Each named attribute on an operation is encoded as:

```
attr_entry {
    name_ref: varint               // String table index for attribute name
    value: attribute               // Attribute value (see Section 4.5)
}
```

### 4.6.4 Region Encoding

A region represents a ordered list of basic blocks. Each basic block contains a sequence of operations.

```
region {
    num_blocks: varint             // Number of basic blocks
    blocks: block[num_blocks]
}

block {
    num_args: varint               // Number of block arguments
    arg_types: varint[num_args]    // Type table indices for arguments
    num_ops: varint                // Number of operations in this block
    operations: operation[num_ops] // Operations
}
```

For regions with a single block (the common case for Tile IR), `num_blocks` is 1 and the single block typically has no arguments (arguments are handled through SSA values from enclosing operations).

**Multi-block regions** are used by control flow operations (e.g., `if` with then/else blocks). Block arguments replace phi nodes for data flow at join points.

### 4.6.5 SSA Value Numbering

Within a function body, SSA values are numbered sequentially starting from 0:

- Function parameters occupy indices 0 through `num_params - 1`.
- Each operation that produces results assigns the next available indices to its results.
- Operands reference these indices.

This numbering is implicit -- there is no explicit index stored for results. The decoder maintains a counter and assigns consecutive indices to each result as operations are decoded.

**Example:** Consider this sequence:

```
entry @example(%a: tile<f32>, %b: tile<f32>) {   // %a = index 0, %b = index 1
    %c = addf %a, %b : tile<f32>                  // %c = index 2
    %d = mulf %c, %a : tile<f32>                  // %d = index 3
    return %d : tile<f32>
}
```

The `addf` operation's operands are encoded as `[0, 1]` and it produces one result (index 2). The `mulf` operation's operands are `[2, 0]` and produces index 3.

### 4.6.6 Opcode Table

Opcodes are assigned as varint values. The complete opcode table is defined by the Tile IR specification. Core opcodes include:

| Opcode | Operation | Operand Pattern | Results |
|--------|-----------|-----------------|---------|
| 0x01 | `module` | 0 operands, 1 region | 0 |
| 0x02 | `entry` | variadic (params) | 0 |
| 0x03 | `return` | variadic (values) | 0 |
| 0x04 | `constant` | 0 operands | 1 |
| 0x05 | `broadcast` | 1 operand | 1 |
| 0x06 | `cat` | variadic | 1 |
| 0x07 | `extract` | 1 operand + index attrs | 1 |
| 0x08 | `get_global` | 0 operands | 1 |
| 0x09 | `iota` | 0 operands | 1 |
| 0x0A | `offset` | 2 operands | 1 |
| 0x0B | `permute` | 1 operand | 1 |
| 0x0C | `reduce` | 2 operands + 1 region | 1 |
| 0x0D | `reshape` | 1 operand | 1 |
| 0x0E | `scan` | 2 operands + 1 region | 1 |
| 0x0F | `select` | 3 operands | 1 |
| 0x10 | `for` | AttrSized segments + 1 region | variadic |
| 0x11 | `if` | 1 condition + 2 regions | variadic |
| 0x12 | `loop` | 0 operands + 1 region | variadic |
| 0x13 | `break` | variadic | 0 |
| 0x14 | `continue` | variadic | 0 |
| 0x15 | `yield` | variadic | 0 |
| 0x16 | `assert` | 1 operand | 0 |
| 0x20 | `addf` | 2 operands | 1 |
| 0x21 | `subf` | 2 operands | 1 |
| 0x22 | `mulf` | 2 operands | 1 |
| 0x23 | `divf` | 2 operands | 1 |
| 0x24 | `mmaf` | 3 operands | 1 |
| 0x25 | `negf` | 1 operand | 1 |
| 0x26 | `absf` | 1 operand | 1 |
| 0x27 | `sqrt` | 1 operand | 1 |
| 0x28 | `rsqrt` | 1 operand | 1 |
| 0x29 | `fma` | 3 operands | 1 |
| 0x30 | `addi` | 2 operands | 1 |
| 0x31 | `subi` | 2 operands | 1 |
| 0x32 | `muli` | 2 operands | 1 |
| 0x33 | `divi` | 2 operands | 1 |
| 0x34 | `mmai` | 3 operands | 1 |
| 0x40 | `load_ptr_tko` | 1 operand | 2 (value + token) |
| 0x41 | `store_ptr_tko` | 2 operands | 1 (token) |
| 0x42 | `make_token` | 0 operands | 1 (token) |
| 0x43 | `join_tokens` | variadic (tokens) | 1 (token) |
| 0x50 | `make_tensor_view` | 1 + variadic | 1 |
| 0x51 | `make_partition_view` | 1 operand | 1 |
| 0x52 | `load_view_tko` | 2 operands | 2 (value + token) |
| 0x53 | `store_view_tko` | 3 operands | 1 (token) |
| 0x54 | `get_index_space_shape` | 1 operand | 1 |
| 0x55 | `get_tensor_shape` | 1 operand | 1 |
| 0x60 | `bitcast` | 1 operand | 1 |
| 0x61 | `exti` | 1 operand | 1 |
| 0x62 | `trunci` | 1 operand | 1 |
| 0x63 | `ftof` | 1 operand | 1 |
| 0x64 | `ftoi` | 1 operand | 1 |
| 0x65 | `itof` | 1 operand | 1 |
| 0x66 | `int_to_ptr` | 1 operand | 1 |
| 0x67 | `ptr_to_int` | 1 operand | 1 |
| 0x68 | `ptr_to_ptr` | 1 operand | 1 |
| 0x70 | `atomic_cas_tko` | 3 operands | 2 (value + token) |
| 0x71 | `atomic_rmw_tko` | 3 operands | 2 (value + token) |

---

## 4.7 Section Ordering

### 4.7.1 Default Writing Order

Serializers should write sections in the following order to enable efficient lazy loading:

```
1. String Section (0x01)       -- Must be first; other sections reference it
2. Type Section (0x05)         -- Second; function table references types
3. Function Table Section (0x02) -- Third; references both strings and types
4. Constant Data Section (0x04) -- Fourth; referenced by attributes and globals
5. Global Section (0x06)       -- Fifth; references strings, types, constants
6. Debug Section (0x03)        -- Last; optional and only needed for debugging
```

This ordering allows a consumer to perform the following incremental loading strategy:

1. Read the String Section and build the string table.
2. Read the Type Section to understand all types in the module.
3. Read the Function Table to identify kernel entry points and their signatures.
4. Optionally stop here if only metadata is needed (e.g., for reflection or linking).
5. Read Constant Data and Globals as needed for full module loading.
6. Read Debug Section only when debug information is requested.

### 4.7.2 Reader Flexibility

The Tile IR format does **not** require sections to appear in the default order. A conforming deserializer must accept sections in any order. To handle arbitrary ordering, the deserializer uses a two-pass strategy:

**Pass 1 (Scan):** Read all section headers, recording the byte offset and length of each section without parsing the section data. This builds a section map.

**Pass 2 (Parse):** Parse sections in dependency order using the section map. If a section references data from another section that has not yet been parsed (e.g., the Function Table references the Type Section), the parser defers resolution until the referenced section is available.

This design accommodates:
- Tools that write sections in arbitrary order.
- Files that have been concatenated or post-processed.
- Future section types with different dependency patterns.

### 4.7.3 Section Dependencies

The dependency graph between sections is:

```
String Section <------+
    |                  |
    v                  |
Type Section           |
    |                  |
    v                  |
Function Table --------+
    |
    v
Constant Data Section
    |
    v
Global Section
    |
    v
Debug Section ----> String Section
```

In this graph, an arrow from A to B means "B depends on A" (B references data within A). The String Section has no dependencies and must be parsable in isolation. The Debug Section depends only on the String Section for file path and name strings.

### 4.7.4 Duplicate Sections

A Tile IR bytecode file must not contain more than one section of any given section ID. If a deserializer encounters a duplicate section ID, it must reject the file as malformed.

### 4.7.5 Missing Required Sections

If a required section (String, Type, or Function Table) is absent, the deserializer must reject the file. The absence of an optional section (Debug, Constant Data, Global) is not an error; the module simply has no debug info, no inline constants, or no globals, respectively.

---

## 4.8 Encoding Examples

This section provides complete encoding examples for small Tile IR programs, demonstrating how the binary format comes together.

### 4.8.1 Minimal Module

Consider the simplest possible Tile IR module:

```
cuda_tile.module @minimal {
    entry @noop_kernel() {
        return
    }
}
```

The binary encoding (in hexadecimal) would be:

```
Magic:
  7F 54 69 6C 65 49 52 00                // \x7FTileIR\x00

Version:
  0D 02 00 00                            // major=13, minor=2, tag=0

--- String Section (ID 0x01) ---
  01                                      // idAndIsAligned: id=1, align=0
  12                                      // length = 18 bytes
  03                                      // 3 strings
  06 6D 69 6E 69 6D 61 6C                 // "minimal" (7 bytes, varint 0x06 + data)
  0C 6E 6F 6F 70 5F 6B 65 72 6E 65 6C 0A  // "noop_kernel" (11 bytes)

--- Type Section (ID 0x05) ---
  05                                      // idAndIsAligned: id=5, align=0
  03                                      // length = 3 bytes
  00                                      // 0 type definitions
  01                                      // op_stream_length = 1
  03                                      // opcode = return (0x03)
                                           // no flags (0x00 implied by compact encoding)

--- Function Table Section (ID 0x02) ---
  02                                      // idAndIsAligned: id=2, align=0
  0B                                      // length = 11 bytes
  01                                      // 1 function entry
  01                                      // name_index = 1 ("noop_kernel")
  01                                      // entry_flag = kernel
  00                                      // num_params = 0
  00                                      // body_offset = 0
  01                                      // body_length = 1
```

### 4.8.2 Vector Add Kernel (128 elements)

```
cuda_tile.module @vec_add {
    entry @vec_add_128(
        %a_ptr: tile<ptr<f32>>,
        %b_ptr: tile<ptr<f32>>,
        %c_ptr: tile<ptr<f32>>
    ) {
        %offset = iota : tile<128xi32>
        %a_base = reshape %a_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
        %a_ptrs = broadcast %a_base : tile<1xptr<f32>> -> tile<128xptr<f32>>
        %a_tensor = offset %a_ptrs, %offset : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
        %a_val, %t1 = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
        %b_val, %t2 = load_ptr_tko weak %b_tensor : ... -> tile<128xf32>, token
        %c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
        store_ptr_tko weak %c_tensor, %c_val : ... -> token
        return
    }
}
```

The type table for this module would contain:

| Index | Type |
|-------|------|
| 0 | F32 (tag 0x07) |
| 1 | I32 (tag 0x03) |
| 2 | ptr\<f32\> (tag 0x0C, element=0) |
| 3 | tile\<ptr\<f32\>\> (tag 0x0D, rank=0, element=2) |
| 4 | tile\<1xptr\<f32\>\> (tag 0x0D, rank=1, shape=[1], element=2) |
| 5 | tile\<128xptr\<f32\>\> (tag 0x0D, rank=1, shape=[128], element=2) |
| 6 | tile\<128xi32\> (tag 0x0D, rank=1, shape=[128], element=1) |
| 7 | tile\<128xf32\> (tag 0x0D, rank=1, shape=[128], element=0) |
| 8 | token (tag 0x11) |

The operation stream for the kernel body encodes each operation with its opcode, flags, result types, attributes, and operands.

### 4.8.3 VarInt Encoding Examples for Common Values

This table shows the byte-level VarInt encoding for values commonly encountered in Tile IR bytecode:

| Value | Description | VarInt Bytes | Length |
|-------|-------------|--------------|--------|
| 0 | Empty count | `00` | 1 |
| 1 | Single item | `01` | 1 |
| 2 | Pair | `02` | 1 |
| 3 | Triple | `03` | 1 |
| 7 | Element type count | `07` | 1 |
| 8 | Pointer tag + F32 | `08` | 1 |
| 13 | Version major | `0D` | 1 |
| 16 | F8E4M3FN tag | `10` | 1 |
| 32 | Small tile dimension | `20` | 1 |
| 64 | Common MMA dimension | `40` | 1 |
| 128 | Vector width | `80 80` | 2 |
| 256 | Extended dimension | `81 00` | 2 |
| 1024 | Block size | `88 00` | 2 |
| 4096 | Large tile | `90 00` | 2 |
| 65536 | Large buffer | `C0 00 00` | 3 |
| 0xFFFFFFFFFFFFFFFF | Dynamic sentinel | `FF FF FF FF FF FF FF FF FF` | 9 |

### 4.8.4 Section Header Examples

**Unaligned String Section with 42 bytes of data:**

```
01                                      // idAndIsAligned: section_id=1 (String), align=0
2A                                      // length = 42 (varint)
<42 bytes of string data>
```

**Aligned Type Section, 8-byte alignment, 1024 bytes of data:**

```
85                                      // idAndIsAligned: section_id=5 (Type), align=1
                                        // Binary: 1_0000101 = 0x85
80 08                                   // length = 1024 (varint)
08                                      // alignment = 8 (varint)
CB CB CB                                // Padding bytes to reach 8-byte alignment
<1024 bytes of type and operation data>
```

**Debug Section with 0 bytes (empty debug info):**

```
03                                      // idAndIsAligned: section_id=3 (Debug), align=0
00                                      // length = 0
```

---

## Appendix A: Summary of Byte Patterns

### A.1 Magic Number

```
7F 54 69 6C 65 49 52 00
```

### A.2 Section ID and Alignment Byte

```
Bit 7:   0 = no alignment, 1 = has alignment
Bits 6-0: section ID (0x01 through 0x06 defined)
```

### A.3 Padding Byte

```
CB = 0xCB (203 decimal)
```

### A.4 Dynamic Dimension Sentinel

```
FF FF FF FF FF FF FF FF FF  (9 bytes, varint for UINT64_MAX)
```

### A.5 Boolean Values

```
0x00 = false
0x01 = true
```

---

## Appendix B: Grammar Summary

This appendix provides a complete grammar for the binary format in BNF-like notation.

```
bytecode       ::= magic version section*
magic          ::= "\x7FTileIR\x00"
version        ::= uint8_major uint8_minor uint16_tag
section        ::= section_header section_padding? section_data
section_header ::= id_and_align_byte length_varint alignment_varint?
id_and_align_byte ::= byte   // bit 7 = has_alignment, bits 6-0 = section_id
length_varint  ::= varint    // byte length of section_data
alignment_varint ::= varint  // present only if has_alignment bit is set
section_padding ::= byte+    // all bytes must be 0xCB
section_data   ::= byte{length}

// Section-specific data
string_section_data     ::= varint_count string_entry*
string_entry            ::= varint_length byte{length}
function_table_data     ::= varint_count function_entry*
function_entry          ::= varint_name_index byte_entry_flag varint_num_params
                             type_ref{num_params} varint_body_offset varint_body_length
type_section_data       ::= varint_type_count type_def* varint_opstream_length byte{opstream_length}
global_section_data     ::= varint_count global_entry*
global_entry            ::= varint_name_index varint_type_ref byte_is_mutable initializer?
constant_data_section   ::= varint_count constant_block*
constant_block          ::= varint_alignment varint_data_length padding byte{data_length}
debug_section_data      ::= varint_version compile_unit* file_table location_table

// Type definitions
type_def       ::= type_tag payload?
type_tag       ::= byte   // 0x00 - 0x11
payload        ::= element_type_payload     // for 0x00-0x0B: empty
               |  pointer_payload           // for 0x0C: varint element_type_ref
               |  tile_payload              // for 0x0D: varint rank varint{rank} varint element_type_ref
               |  tensor_view_payload       // for 0x0E: varint rank varint{rank} varint element_ref varint stride_count varint{stride_count}
               |  partition_view_payload    // for 0x0F: varint tile_rank varint{tile_rank} varint tv_ref varint dim_count varint{dim_count}
               |  function_payload          // for 0x10: varint num_params varint{num_params} varint num_results varint{num_results}

// Attributes
attribute      ::= attr_tag attr_payload?
attr_tag       ::= byte   // 0x01 - 0x0C
attr_payload   ::= integer_attr_payload
               |  float_attr_payload
               |  bool_attr_payload
               |  type_attr_payload
               |  string_attr_payload
               |  array_attr_payload
               |  dense_elements_payload
               |  div_by_payload
               |  same_elements_payload
               |  dictionary_attr_payload
               |  optimization_hints_payload
               |  empty_payload

// Operations
operation      ::= varint_opcode varint_location_index byte_flags
                   result_types? attributes? operands regions?
result_types   ::= varint_count varint{count}   // present if has_results flag
attributes     ::= varint_count attr_entry{count}  // present if has_attrs flag
attr_entry     ::= varint_name_ref attribute
operands       ::= fixed_operands | variadic_operands | attr_sized_operands
regions        ::= varint_count region{count}    // present if has_regions flag
region         ::= varint_num_blocks block{num_blocks}
block          ::= varint_num_args varint{num_args} varint_num_ops operation{num_ops}
```

---

## Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| 13.0 | 2025-06 | Initial public release of Tile IR binary format |
| 13.1 | 2025-10 | Added F8E4M3FN and F8E5M2 type tags (0x0A, 0x0B). Added atomics sections. Enhanced debug section with inline scope support. |
| 13.2 | 2026-03 | Added OptimizationHints attribute (0x0B). NonNegative attribute (0x0C). PartitionView dim_map encoding. Expanded opcode range for view operations. |

---

*End of Chapter 4: Binary Format and Bytecode*
