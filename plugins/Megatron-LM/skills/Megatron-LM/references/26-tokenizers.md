# Chapter 26: Tokenizers

## Source Files
- `megatron/core/tokenizers/base_tokenizer.py` - MegatronTokenizerBase abstract class
- `megatron/core/tokenizers/megatron_tokenizer.py` - MegatronTokenizer main class
- `megatron/core/tokenizers/text/text_tokenizer.py` - Text tokenizer base
- `megatron/core/tokenizers/text/models/gpt_tokenizer.py` - GPT BPE tokenizer
- `megatron/core/tokenizers/text/models/default_tokenizer.py` - Default tokenizer
- `megatron/core/tokenizers/text/libraries/abstract_tokenizer.py` - Abstract tokenizer interface
- `megatron/core/tokenizers/text/libraries/huggingface_tokenizer.py` - HuggingFace integration
- `megatron/core/tokenizers/text/libraries/sentencepiece_tokenizer.py` - SentencePiece
- `megatron/core/tokenizers/text/libraries/tiktoken_tokenizer.py` - TikToken
- `megatron/core/tokenizers/text/libraries/megatron_hf_tokenizer.py` - Megatron HF tokenizer
- `megatron/core/tokenizers/text/libraries/sft_tokenizer.py` - SFT tokenizer
- `megatron/core/tokenizers/text/libraries/chat_template.py` - Chat template support
- `megatron/core/tokenizers/text/libraries/bytelevel_tokenizer.py` - Byte-level tokenizer
- `megatron/core/tokenizers/text/libraries/null_tokenizer.py` - Null tokenizer
- `megatron/core/tokenizers/vision/vision_tokenizer.py` - Vision tokenizer
- `megatron/core/tokenizers/utils/build_tokenizer.py` - Tokenizer builder function

## Tokenizer Architecture

### Class Hierarchy

```
MegatronTokenizerBase (ABC)
├── tokenize()           # Encode text to token IDs
├── detokenize()         # Decode token IDs to text
├── vocab()              # Get vocabulary
├── vocab_size()         # Get vocabulary size
└── apply_chat_template() # Apply chat template

MegatronTokenizer (from_pretrained)
├── TextTokenizer
│   ├── GPTTokenizer (GPT2BPE)
│   ├── DefaultTokenizer
│   └── HuggingFace-backed tokenizers
├── VisionTokenizer
└── NullTokenizer / NullMultimodalTokenizer
```

## Supported Tokenizers

### 1. BertWordPieceLowerCase / BertWordPieceCase

WordPiece tokenizer from BERT, used for BERT-style models:

```bash
--tokenizer-type BertWordPieceCase
--vocab-file /path/to/vocab.txt
```

Characteristics:
- Case-sensitive (`BertWordPieceCase`) or lowercase (`BertWordPieceLowerCase`)
- WordPiece tokenization algorithm
- Fixed vocabulary file
- Includes 100 additional special tokens (`<extra_id_0>` through `<extra_id_99>`)

### 2. GPT2BPETokenizer

Byte Pair Encoding tokenizer used for GPT-2 style models:

```bash
--tokenizer-type GPT2BPETokenizer
--vocab-file /path/to/vocab.json
--merge-file /path/to/merges.txt
```

Characteristics:
- Byte-level BPE
- Requires both vocabulary file and merge rules file
- Compatible with OpenAI GPT-2 vocabulary format

### 3. SentencePieceTokenizer

Google SentencePiece tokenizer:

```bash
--tokenizer-type SentencePieceTokenizer
--tokenizer-model /path/to/tokenizer.model
--tokenizer-sentencepiece-legacy   # Use legacy mode
```

Characteristics:
- Supports both BPE and Unigram modes
- Single model file
- Optional legacy mode for backward compatibility
- Used by Llama 2 and similar models

### 4. GPTSentencePieceTokenizer

GPT variant of SentencePiece:

```bash
--tokenizer-type GPTSentencePieceTokenizer
--tokenizer-model /path/to/tokenizer.model
```

### 5. Llama2Tokenizer

Llama 2 specific SentencePiece variant:

```bash
--tokenizer-type Llama2Tokenizer
--tokenizer-model /path/to/tokenizer.model
```

### 6. TikTokenizer

OpenAI's TikToken tokenizer (used by GPT-4, etc.):

```bash
--tokenizer-type TikTokenizer
--tokenizer-model /path/to/tokenizer.model
--tiktoken-pattern "some_pattern"
--tiktoken-num-special-tokens 100
--special-tokens "<|end|>" "<|pad|>"
```

Characteristics:
- Efficient Rust-based implementation
- Custom pattern support for tokenization rules
- Special token handling
- Compatible with cl100k_base and o200k_base encodings

### 7. HuggingFaceTokenizer

Universal HuggingFace tokenizer wrapper supporting all HF tokenizers:

```bash
--tokenizer-type HuggingFaceTokenizer
--tokenizer-model /path/to/tokenizer
--trust-remote-code                 # Allow custom tokenizer code
--tokenizer-hf-no-use-fast          # Disable fast tokenizer
--tokenizer-hf-no-include-special-tokens  # Exclude special tokens
```

Characteristics:
- Supports all HuggingFace tokenizers (LlamaTokenizer, CodeLlama, etc.)
- Fast tokenizer by default (Rust-based)
- Optional trust_remote_code for custom tokenizer implementations
- Can load from local path or HuggingFace Hub ID
- Supports chat templates via the HF tokenizer

### 8. MegatronHuggingFaceTokenizer

Enhanced HuggingFace tokenizer with Megatron-specific features:

```bash
--tokenizer-type HuggingFaceTokenizer
--tokenizer-model /path/to/tokenizer
```

Adds features beyond standard HF tokenizer:
- Chat template support
- Special token management
- Metadata handling for Megatron integration

### 9. SFTTokenizer

Tokenizer for supervised fine-tuning with prompt formatting:

```bash
--tokenizer-type SFTTokenizer
--tokenizer-model /path/to/tokenizer
--sft-tokenizer-prompt-format "chatml"  # or "alpaca", "raw", etc.
```

Supports prompt formats:
- `chatml`: ChatML format with `<|im_start|>` and `<|im_end|>` tokens
- `alpaca`: Alpaca instruction format
- `raw`: No formatting, raw text

### 10. NullTokenizer

Passthrough tokenizer that does not perform any tokenization:

```bash
--tokenizer-type NullTokenizer
--vocab-size 32000
--null-tokenizer-eod-id 1           # Custom EOD token ID
--null-tokenizer-pad-id 0           # Custom pad token ID
```

Used for:
- Pre-tokenized data that does not need further tokenization
- Debugging and testing
- Custom tokenization pipelines

### 11. NullMultimodalTokenizer

Null tokenizer variant for multimodal data:

```bash
--tokenizer-type NullMultimodalTokenizer
--vocab-size 32000
```

### 12. MultimodalTokenizer

Tokenizer for vision-language models:

```bash
--tokenizer-type MultimodalTokenizer
--tokenizer-model /path/to/tokenizer
--tokenizer-prompt-format "chatml"
--image-tag-type "interleave"        # Image tag handling
--force-system-message               # Force system message
--special-tokens "<image>" "</image>"
```

## Tokenizer Builder

The `build_tokenizer` function in `build_tokenizer.py` creates the appropriate tokenizer based on `args.tokenizer_type`:

```python
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

tokenizer = build_tokenizer(args)
```

### Builder Logic

1. Check `tokenizer_type` against known tokenizer categories:
   - `MEGATRON_TOKENIZERS`: BertWordPieceLowerCase, BertWordPieceCase, GPT2BPETokenizer
   - `SP_TOKENIZERS`: SentencePieceTokenizer, GPTSentencePieceTokenizer, Llama2Tokenizer
   - TikTokenizer
   - HuggingFaceTokenizer
   - MultimodalTokenizer
   - SFTTokenizer
   - NullTokenizer / NullMultimodalTokenizer

2. Determine the tokenizer library:
   - `megatron`: Megatron-native tokenizers
   - `sentencepiece`: SentencePiece-based tokenizers
   - `tiktoken`: TikToken-based tokenizers
   - `huggingface`: HuggingFace tokenizers
   - `sft`: SFT tokenizers
   - `null-text` / `null-multimodal`: Null tokenizers

3. Create the tokenizer via `MegatronTokenizer.from_pretrained()`:
   ```python
   tokenizer = MegatronTokenizer.from_pretrained(
       tokenizer_path=tokenizer_path,
       metadata_path={'library': tokenizer_library},
       **kwargs
   )
   ```

4. Set the padded vocab size:
   ```python
   args.padded_vocab_size = vocab_size_with_padding(tokenizer.vocab_size, args)
   ```

### Vocabulary Padding

The vocabulary size is padded to be divisible by both `make_vocab_size_divisible_by` and `tensor_model_parallel_size`:

```python
def vocab_size_with_padding(orig_vocab_size, args):
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    padded = int(math.ceil(orig_vocab_size / multiple) * multiple)
    return padded
```

This ensures:
- The embedding table can be evenly split across tensor-parallel ranks
- The padded tokens are initialized as zeros and never updated

## Special Tokens

### Configuring Special Tokens

```bash
--special-tokens "<|endoftext|>" "<|end|>" "<|pad|>"
```

Special tokens are passed to the tokenizer during construction and are:
- Added to the vocabulary
- Never split into sub-tokens
- Used for specific purposes (EOD, padding, etc.)

### End-of-Document (EOD) Token

The EOD token marks document boundaries:
- Added during preprocessing with `--append-eod`
- Used by GPT dataset to handle multi-document sequences
- Can be configured via `--eod-mask-loss`

### Chat Template Tokens

For chat/instruction models, special tokens mark conversation turns:

```bash
--special-tokens "<|im_start|>" "<|im_end|>"
```

The `apply_chat_template` method handles formatting:
```python
formatted = tokenizer.apply_chat_template(messages)
# messages = [
#     {"role": "system", "content": "You are helpful."},
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hi there!"},
# ]
```

## Vision Tokenizers

Vision tokenizers handle image tokenization for multimodal models:

```python
from megatron.core.tokenizers.vision.vision_tokenizer import VisionTokenizer
```

Vision tokenizers:
- Convert images to sequences of visual tokens
- Interface with vision encoders (CLIP, SigLIP, etc.)
- Support different image resolution and patch sizes
- Handle image-text interleaving with special tokens

## Tokenizer Metadata

Tokenizers can store metadata in a metadata file:

```bash
--metadata-path /path/to/metadata.json
```

The metadata includes:
- `library`: Tokenizer library type
- `class_name`: Tokenizer class name
- `class_path`: Path to tokenizer class
- `model_type`: Model type the tokenizer is designed for
- `chat_template`: Chat template string

## Configuration Examples

### Llama 3 Style
```bash
--tokenizer-type HuggingFaceTokenizer
--tokenizer-model meta-llama/Meta-Llama-3-8B
--trust-remote-code
```

### GPT-2 Style
```bash
--tokenizer-type GPT2BPETokenizer
--vocab-file gpt2-vocab.json
--merge-file gpt2-merges.txt
```

### TikToken (GPT-4 Style)
```bash
--tokenizer-type TikTokenizer
--tokenizer-model /path/to/tiktoken.model
--tiktoken-num-special-tokens 100
```

### Multimodal (LLaVA Style)
```bash
--tokenizer-type MultimodalTokenizer
--tokenizer-model /path/to/tokenizer
--special-tokens "<image>" "</image>"
--image-tag-type interleave
```

### Pre-tokenized Data
```bash
--tokenizer-type NullTokenizer
--vocab-size 32000
```

### SFT with Chat Templates
```bash
--tokenizer-type SFTTokenizer
--tokenizer-model /path/to/tokenizer
--sft-tokenizer-prompt-format chatml
```
