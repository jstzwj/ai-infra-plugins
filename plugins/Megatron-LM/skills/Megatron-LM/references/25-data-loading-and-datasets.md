# Chapter 25: Data Loading and Datasets

## Source Files
- `megatron/core/datasets/gpt_dataset.py` - GPT dataset implementation
- `megatron/core/datasets/bert_dataset.py` - BERT dataset implementation
- `megatron/core/datasets/t5_dataset.py` - T5 dataset implementation
- `megatron/core/datasets/masked_dataset.py` - Masked language modeling datasets
- `megatron/core/datasets/multimodal_dataset.py` - Multimodal datasets
- `megatron/core/datasets/blended_dataset.py` - Blended dataset mixing
- `megatron/core/datasets/blended_megatron_dataset_builder.py` - Builder for blended datasets
- `megatron/core/datasets/blended_megatron_dataset_config.py` - Blended dataset configuration
- `megatron/core/datasets/indexed_dataset.py` - Binary indexed dataset format
- `megatron/core/datasets/megatron_dataset.py` - Base MegatronDataset class
- `megatron/core/datasets/data_schedule.py` - Data scheduling (HybridCPDataLoaderWrapper)
- `megatron/core/datasets/helpers.py` - Dataset helper functions
- `megatron/core/datasets/utils.py` - Dataset utilities
- `tools/preprocess_data.py` - Data preprocessing tool
- `docs/user-guide/data-preparation.md` - Data preparation docs

## Data Format

### JSONL Input Format

Megatron-LM expects training data in JSONL (JSON Lines) format, where each line is a JSON object:

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
{"text": "More training data..."}
```

For multimodal data, additional fields are supported:
```json
{"text": "Describe this image.", "image": "/path/to/image.jpg"}
```

For fill-in-the-middle (FIM) training:
```json
{"text": "def hello_world():\n    print('Hello')\n    return True"}
```

## Preprocessing

### preprocess_data.py

Converts JSONL data to Megatron's binary indexed format:

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--input` | Path to input JSON/JSONL file |
| `--output-prefix` | Prefix for output binary files (.bin and .idx) |
| `--tokenizer-type` | Tokenizer type (e.g., HuggingFaceTokenizer, GPT2BPETokenizer) |
| `--tokenizer-model` | Path to tokenizer model file |
| `--vocab-file` | Path to vocabulary file |
| `--merge-file` | Path to BPE merge file |
| `--workers` | Number of parallel workers |
| `--append-eod` | Append end-of-document token |
| `--log-interval` | Logging interval |
| `--keep-sequential-samples` | Keep samples in sequential order |

### Optimal Worker Count

Find the best number of workers automatically:

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --find-optimal-num-workers \
    --workers-to-check 4 8 16 32 \
    --max-documents 50000
```

### Output Files

Preprocessing generates two files:
- `processed_data.bin` - Binary file containing tokenized sequences
- `processed_data.idx` - Index file for random access

## Binary Indexed Format

The `IndexedDataset` class provides memory-mapped access to preprocessed data:

### Format Structure

```
IndexedDataset:
  .idx file:
    Header: "MMIDIDX\x00\x00" (magic number)
    Version, dtype code, number of documents, number of tokens
    Document sizes array
    Document indices (pointers into .bin file)
  .bin file:
    Raw token IDs in the specified dtype
```

### Supported Token Dtypes

| DType | Code | Use Case |
|------|------|----------|
| `uint8` | 1 | Vocab size <= 256 |
| `int8` | 2 | Signed small vocab |
| `int16` | 3 | Vocab size <= 32768 |
| `int32` | 4 | Standard (vocab size <= 2^31) |
| `int64` | 5 | Large vocab |
| `float64` | 6 | Embedding data |
| `float32` | 7 | Embedding data |
| `uint16` | 8 | Vocab size <= 65536 |

The token dtype is automatically selected based on vocab size:
- `uint16` when `vocab_size <= 65535`
- `int32` when `vocab_size > 65535`

### Memory Mapping

IndexedDataset uses memory mapping (mmap) for efficient access:
- No need to load the entire dataset into memory
- Random access by document index is O(1)
- Multiple processes can share the same memory-mapped file

### Object Storage Support

IndexedDataset supports loading from S3 and other object storage:

```bash
--data-path s3://bucket/path/to/dataset
```

Or with the MultiStorageClient (MSC):
```bash
--data-path msc://profile/path/to/dataset
```

Object storage datasets are automatically cached locally.

## Dataset Types

### GPT Dataset

Autoregressive language modeling dataset. Each sample contains a sequence of tokens for next-token prediction.

```python
GPTDatasetConfig(
    reset_position_ids=True,      # Reset position IDs at document boundaries
    reset_attention_mask=True,     # Reset attention mask at document boundaries
    eod_mask_loss=False,          # Mask loss at EOD tokens
    create_attention_mask=True,    # Generate attention masks
    add_extra_token_to_sequence=True,  # Extra token for input/output alignment
)
```

**Features:**
- Document boundary handling with EOD tokens
- Optional position ID reset between documents
- Optional attention mask reset (causal attention within documents)
- Input tokens = tokens[:-1], output tokens = tokens[1:]

### BERT Dataset

Masked language modeling dataset for encoder-only models:

```python
from megatron.core.datasets.bert_dataset import BertDataset
```

Features:
- Masked token prediction (randomly mask 15% of tokens)
- Next sentence prediction (NSP)
- Segment embedding support

### T5 Dataset

Encoder-decoder dataset for sequence-to-sequence models:

```python
from megatron.core.datasets.t5_dataset import T5Dataset
```

Features:
- Encoder input with sentinel tokens
- Decoder target with sentinel tokens
- Span corruption for pretraining

### Multimodal Dataset

Dataset for vision-language models:

```python
from megatron.core.datasets.multimodal_dataset import MultimodalDataset
```

Features:
- Text and image pair loading
- Image preprocessing and augmentation
- Vision token embedding support

## Blended Datasets

Combine multiple datasets with specified weights:

### Configuration

```python
BlendedMegatronDatasetConfig(
    random_seed=1234,
    sequence_length=2048,
    blend="dataset1:0.5 dataset2:0.3 dataset3:0.2",
    split="969,30,1",           # Train/valid/test split percentages
)
```

### Blend Specification

The `--data-path` argument supports blend specifications:

```bash
# Single dataset
--data-path /path/to/dataset

# Blended with weights (proportional sampling)
--data-path 0.5 /path/to/dataset1 0.3 /path/to/dataset2 0.2 /path/to/dataset3

# Blended with per-split weights
--train-data-path 0.5 dataset1 0.5 dataset2
--valid-data-path 1.0 dataset3
--test-data-path 1.0 dataset4
```

### BlendedDataset Implementation

The `BlendedDataset` class:
1. Takes a list of `MegatronDataset` instances and their weights
2. Normalizes weights to sum to 1.0
3. Creates a permutation that maps each sample index to a (dataset, internal_index) pair
4. On `__getitem__`, delegates to the appropriate underlying dataset

The blending is deterministic based on the random seed, ensuring reproducibility.

### Per-Dataset Sequence Control

Control the number of sequences drawn from each dataset:

```bash
--per-dataset-sequences-path sequences.json
```

The JSON file maps dataset names to sequence counts:
```json
{
    "dataset1": 1000000,
    "dataset2": 500000,
    "dataset3": 200000
}
```

## Fill-In-the-Middle (FIM)

FIM training augments code data with span prediction tasks:

```bash
--fim-rate 0.5                 # Probability of applying FIM
--fim-spm-rate 0.5             # SPM vs PSM split rate
--fim-prefix-len 50            # Max prefix length
--fim-middle-len 100           # Max middle length
--fim-suffix-len 50            # Max suffix length
```

FIM transforms a code sample into:
- Prefix-Suffix-Middle (PSM): `<pre> prefix <suf> suffix <mid> middle`
- Suffix-Prefix-Middle (SPM): `<suf> suffix <pre> prefix <mid> middle`

## Sequence Packing

Pack multiple short sequences into a single training sample for efficiency:

```bash
--seq-length 4096
--pack-sequences                # Enable sequence packing
```

When enabled:
- Multiple documents are packed into a single sequence of `seq_length` tokens
- Position IDs and attention masks correctly handle document boundaries
- Training efficiency improves significantly for datasets with variable-length documents

### Packing with Hybrid Context Parallelism

The `HybridCPDataLoaderWrapper` handles sequence packing with hybrid context parallelism:

```python
class HybridCPDataLoaderWrapper:
    """Wraps a data_iterator to distribute packed sequences across DPxCP ranks."""
```

It:
1. Pulls a batch of packed samples from the data iterator
2. Extracts sequence lengths of each sub-sample
3. All-gathers sequence lengths across the DP group
4. Schedules sub-samples to DPxCP ranks using `BalancedCPScheduler`
5. Routes sub-samples via all-to-all based on the schedule

## Data Scheduling

### Data Samplers

Megatron uses distributed data samplers that ensure:
- Each DP rank gets a unique subset of the data
- Epoch boundaries are handled correctly
- Deterministic ordering based on seed

```python
from megatron.training.datasets.data_samplers import build_pretraining_data_loader
```

### Hybrid CP Data Loader

For hybrid context parallelism:

```python
from megatron.core.datasets.data_schedule import HybridCPDataLoaderWrapper

data_loader = HybridCPDataLoaderWrapper(
    data_iterator=base_iterator,
    config=config,
    pg_collection=pg_collection,
)
```

The balanced CP scheduler ensures:
- Equal computation across CP ranks
- Minimal padding waste
- Deterministic assignment based on sequence lengths

## Megatron Energon

Megatron Energon is a data loading framework for large-scale multimodal training. It provides:
- Efficient data loading from object storage
- On-the-fly data augmentation
- Multi-modal data mixing
- Integration with Megatron's distributed training

## Data Loading Pipeline Summary

```
JSONL files
    │
    ▼
preprocess_data.py (tokenization + binary format)
    │
    ▼
IndexedDataset (.bin + .idx)
    │
    ▼
BlendedMegatronDatasetBuilder
    ├── BlendedMegatronDatasetConfig
    ├── GPTDatasetConfig / BERTDatasetConfig / T5DatasetConfig
    └── Tokenizer
    │
    ▼
MegatronDataset (GPTDataset, BERTDataset, etc.)
    │
    ▼ (optional)
BlendedDataset (mix multiple MegatronDatasets)
    │
    ▼
build_pretraining_data_loader()
    │
    ▼
HybridCPDataLoaderWrapper (optional, for hybrid CP)
    │
    ▼
Training loop
```

## Configuration Examples

### Basic GPT Training
```bash
--data-path /data/my_dataset_text_document
--tokenizer-type HuggingFaceTokenizer
--tokenizer-model /path/to/tokenizer
--seq-length 2048
```

### Blended Dataset Training
```bash
--data-path 0.7 /data/code_dataset 0.3 /data/prose_dataset
--split 98,1,1
```

### Sequence Packing
```bash
--data-path /data/my_dataset_text_document
--seq-length 4096
--pack-sequences
```

### S3 Data Loading
```bash
--data-path s3://my-bucket/datasets/my_dataset_text_document
--object-storage-cache-path /tmp/cache
```

### Multimodal Data
```bash
--data-path /data/multimodal_dataset
--tokenizer-type MultimodalTokenizer
--tokenizer-model /path/to/tokenizer
--seq-length 2048
```
