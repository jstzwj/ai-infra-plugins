# vLLM Supported Models Reference

This document provides a comprehensive catalog of all model architectures supported by vLLM, along with the model registry system, configuration handling, and capability interfaces.

---

## Table of Contents

1. [Model Registry System](#model-registry-system)
2. [Model Categories](#model-categories)
3. [Text Generation Models](#text-generation-models)
4. [Embedding Models](#embedding-models)
5. [Late Interaction Models](#late-interaction-models)
6. [Reward Models](#reward-models)
7. [Token Classification Models](#token-classification-models)
8. [Sequence Classification Models](#sequence-classification-models)
9. [Multimodal Models](#multimodal-models)
10. [Speculative Decoding Models](#speculative-decoding-models)
11. [Transformers Backend Models](#transformers-backend-models)
12. [Previously Supported Models](#previously-supported-models)
13. [Out-of-Tree Plugin Models](#out-of-tree-plugin-models)
14. [Model Configuration](#model-configuration-modelconfig)
15. [Model Architecture Config](#model-architecture-config)
16. [Model-Specific Configuration Handlers](#model-specific-configuration-handlers)
17. [Model Capability Interfaces](#model-capability-interfaces)
18. [Model Info Dataclass](#model-info-dataclass)

---

## Model Registry System

**Source:** `vllm/model_executor/models/registry.py`

The model registry is vLLM's central system for discovering, loading, and querying model architectures. It maps HuggingFace architecture names to vLLM implementation modules and classes.

### Architecture

The registry uses a lazy-loading pattern to avoid importing all model implementations at startup. Models are registered as `_LazyRegisteredModel` entries that only import the actual model class when needed. This prevents CUDA initialization errors in subprocess-based model inspection.

### Key Classes

#### `_ModelRegistry`

```python
@dataclass
class _ModelRegistry:
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)
```

The global singleton `ModelRegistry` is created from all model category dictionaries at module load time.

**Methods:**

- `get_supported_archs() -> Set[str]` - Returns all registered architecture names.

- `register_model(model_arch: str, model_cls: type[nn.Module] | str) -> None` - Register an external model. `model_cls` can be either a `torch.nn.Module` class directly, or a string in `<module>:<class>` format for lazy loading.

- `inspect_model_cls(architectures: str | list[str], model_config: ModelConfig) -> tuple[_ModelInfo, str]` - Inspect model capabilities without loading the full class. Returns `_ModelInfo` and the matched architecture name. Resolution order:
  1. If `model_impl == "transformers"`, resolve via transformers backend
  2. If `model_impl == "terratorch"`, resolve to `"Terratorch"`
  3. Try normalized architecture matching with runner/convert type defaults
  4. Fallback to transformers backend for `model_impl == "auto"`

- `resolve_model_cls(architectures: str | list[str], model_config: ModelConfig) -> tuple[type[nn.Module], str]` - Load and return the actual model class. Same resolution logic as `inspect_model_cls`.

- `is_text_generation_model(architectures, model_config) -> bool`
- `is_pooling_model(architectures, model_config) -> bool`
- `is_multimodal_model(architectures, model_config) -> bool`
- `is_multimodal_raw_input_only_model(architectures, model_config) -> bool`
- `is_pp_supported_model(architectures, model_config) -> bool`
- `model_has_inner_state(architectures, model_config) -> bool`
- `is_attention_free_model(architectures, model_config) -> bool`
- `is_hybrid_model(architectures, model_config) -> bool`
- `is_noops_model(architectures, model_config) -> bool`
- `is_transcription_model(architectures, model_config) -> bool`
- `is_transcription_only_model(architectures, model_config) -> bool`

#### `_LazyRegisteredModel`

```python
@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    module_name: str
    class_name: str
```

Lazy-loaded model registration. Key features:
- **Model info caching:** Inspects are cached as JSON files under `$VLLM_CACHE_ROOT/modelinfos/` keyed by module content hash.
- **Subprocess inspection:** Model class inspection runs in a subprocess to avoid CUDA initialization in the main process.
- **Hash-based invalidation:** Cache is invalidated when the source file's content hash changes.

**Methods:**
- `inspect_model_cls() -> _ModelInfo` - Returns model capabilities. Uses cache if available, otherwise runs subprocess inspection.
- `load_model_cls() -> type[nn.Module]` - Imports and returns the model class via `importlib.import_module`.

#### `_RegisteredModel`

```python
@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    interfaces: _ModelInfo
    model_cls: type[nn.Module]
```

Eagerly-loaded model registration (used when a model class is passed directly).

### Registration Format

All models in the registry dictionaries use the format:
```python
"ArchitectureName": ("module_name", "ClassName")
```

For example:
```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM")
```

This maps to the module `vllm.model_executor.models.llama` and class `LlamaForCausalLM`.

### Subprocess Inspection

Model inspection (determining capabilities without loading CUDA) uses a subprocess mechanism:

```python
_SUBPROCESS_COMMAND = [sys.executable, "-m", "vllm.model_executor.models.registry"]
```

The function `_run_in_subprocess(fn)` pickles the inspection function with `cloudpickle`, sends it to a subprocess via stdin, and reads back the result.

---

## Model Categories

vLLM organizes models into the following categories:

| Category | Dictionary | Approximate Count |
|----------|-----------|-------------------|
| Text Generation | `_TEXT_GENERATION_MODELS` | ~150 |
| Embedding | `_EMBEDDING_MODELS` | ~30 |
| Late Interaction | `_LATE_INTERACTION_MODELS` | ~12 |
| Reward | `_REWARD_MODELS` | 3 |
| Token Classification | `_TOKEN_CLASSIFICATION_MODELS` | 4 |
| Sequence Classification | `_SEQUENCE_CLASSIFICATION_MODELS` | ~12 |
| Multimodal | `_MULTIMODAL_MODELS` | ~80 |
| Speculative Decoding | `_SPECULATIVE_DECODING_MODELS` | ~40 |
| Transformers Supported | `_TRANSFORMERS_SUPPORTED_MODELS` | 2 |
| Transformers Backend | `_TRANSFORMERS_BACKEND_MODELS` | 10 |

All categories are merged into `_VLLM_MODELS` which feeds the global `ModelRegistry`.

---

## Text Generation Models

**Source:** `_TEXT_GENERATION_MODELS` in `registry.py`

Decoder-only causal language models for text generation.

### Complete Architecture Catalog

| Architecture | Module | Class |
|-------------|--------|-------|
| `AfmoeForCausalLM` | `afmoe` | `AfmoeForCausalLM` |
| `ApertusForCausalLM` | `apertus` | `ApertusForCausalLM` |
| `AquilaModel` | `llama` | `LlamaForCausalLM` |
| `AquilaForCausalLM` | `llama` | `LlamaForCausalLM` |
| `ArceeForCausalLM` | `arcee` | `ArceeForCausalLM` |
| `ArcticForCausalLM` | `arctic` | `ArcticForCausalLM` |
| `AXK1ForCausalLM` | `AXK1` | `AXK1ForCausalLM` |
| `BaiChuanForCausalLM` | `baichuan` | `BaiChuanForCausalLM` |
| `BaichuanForCausalLM` | `baichuan` | `BaichuanForCausalLM` |
| `BailingMoeForCausalLM` | `bailing_moe` | `BailingMoeForCausalLM` |
| `BailingMoeV2ForCausalLM` | `bailing_moe` | `BailingMoeV2ForCausalLM` |
| `BailingMoeV2_5ForCausalLM` | `bailing_moe_linear` | `BailingMoeV25ForCausalLM` |
| `BambaForCausalLM` | `bamba` | `BambaForCausalLM` |
| `BloomForCausalLM` | `bloom` | `BloomForCausalLM` |
| `ChatGLMModel` | `chatglm` | `ChatGLMForCausalLM` |
| `ChatGLMForConditionalGeneration` | `chatglm` | `ChatGLMForCausalLM` |
| `CohereForCausalLM` | `commandr` | `CohereForCausalLM` |
| `Cohere2ForCausalLM` | `commandr` | `CohereForCausalLM` |
| `CohereMoeForCausalLM` | `cohere_moe` | `CohereMoeForCausalLM` |
| `CwmForCausalLM` | `llama` | `LlamaForCausalLM` |
| `DbrxForCausalLM` | `dbrx` | `DbrxForCausalLM` |
| `DeciLMForCausalLM` | `nemotron_nas` | `DeciLMForCausalLM` |
| `DeepseekForCausalLM` | `deepseek_v2` | `DeepseekForCausalLM` |
| `DeepseekV2ForCausalLM` | `deepseek_v2` | `DeepseekV2ForCausalLM` |
| `DeepseekV3ForCausalLM` | `deepseek_v2` | `DeepseekV3ForCausalLM` |
| `DeepseekV32ForCausalLM` | `deepseek_v2` | `DeepseekV3ForCausalLM` |
| `DeepseekV4ForCausalLM` | `deepseek_v4` | `DeepseekV4ForCausalLM` |
| `Dots1ForCausalLM` | `dots1` | `Dots1ForCausalLM` |
| `Ernie4_5ForCausalLM` | `ernie45` | `Ernie4_5ForCausalLM` |
| `Ernie4_5_MoeForCausalLM` | `ernie45_moe` | `Ernie4_5_MoeForCausalLM` |
| `ExaoneForCausalLM` | `exaone` | `ExaoneForCausalLM` |
| `Exaone4ForCausalLM` | `exaone4` | `Exaone4ForCausalLM` |
| `ExaoneMoEForCausalLM` | `exaone_moe` | `ExaoneMoeForCausalLM` |
| `Fairseq2LlamaForCausalLM` | `fairseq2_llama` | `Fairseq2LlamaForCausalLM` |
| `FalconForCausalLM` | `falcon` | `FalconForCausalLM` |
| `FalconMambaForCausalLM` | `mamba` | `MambaForCausalLM` |
| `FalconH1ForCausalLM` | `falcon_h1` | `FalconH1ForCausalLM` |
| `FlexOlmoForCausalLM` | `flex_olmo` | `FlexOlmoForCausalLM` |
| `GemmaForCausalLM` | `gemma` | `GemmaForCausalLM` |
| `Gemma2ForCausalLM` | `gemma2` | `Gemma2ForCausalLM` |
| `Gemma3ForCausalLM` | `gemma3` | `Gemma3ForCausalLM` |
| `Rnj1ForCausalLM` | `rnj1` | `Rnj1ForCausalLM` |
| `Gemma3nForCausalLM` | `gemma3n` | `Gemma3nForCausalLM` |
| `Gemma4ForCausalLM` | `gemma4` | `Gemma4ForCausalLM` |
| `Qwen3NextForCausalLM` | `qwen3_next` | `Qwen3NextForCausalLM` |
| `GlmForCausalLM` | `glm` | `GlmForCausalLM` |
| `Glm4ForCausalLM` | `glm4` | `Glm4ForCausalLM` |
| `Glm4MoeForCausalLM` | `glm4_moe` | `Glm4MoeForCausalLM` |
| `Glm4MoeLiteForCausalLM` | `glm4_moe_lite` | `Glm4MoeLiteForCausalLM` |
| `GlmMoeDsaForCausalLM` | `deepseek_v2` | `GlmMoeDsaForCausalLM` |
| `GptOssForCausalLM` | `gpt_oss` | `GptOssForCausalLM` |
| `GPT2LMHeadModel` | `gpt2` | `GPT2LMHeadModel` |
| `GPTBigCodeForCausalLM` | `gpt_bigcode` | `GPTBigCodeForCausalLM` |
| `GPTJForCausalLM` | `gpt_j` | `GPTJForCausalLM` |
| `GPTNeoXForCausalLM` | `gpt_neox` | `GPTNeoXForCausalLM` |
| `GraniteForCausalLM` | `granite` | `GraniteForCausalLM` |
| `GraniteMoeForCausalLM` | `granitemoe` | `GraniteMoeForCausalLM` |
| `GraniteMoeHybridForCausalLM` | `granitemoehybrid` | `GraniteMoeHybridForCausalLM` |
| `GraniteMoeSharedForCausalLM` | `granitemoeshared` | `GraniteMoeSharedForCausalLM` |
| `GritLM` | `gritlm` | `GritLM` |
| `Grok1ModelForCausalLM` | `grok1` | `GrokForCausalLM` |
| `Grok1ForCausalLM` | `grok1` | `GrokForCausalLM` |
| `HunYuanMoEV1ForCausalLM` | `hunyuan_v1` | `HunYuanMoEV1ForCausalLM` |
| `HunYuanDenseV1ForCausalLM` | `hunyuan_v1` | `HunYuanDenseV1ForCausalLM` |
| `HYV3ForCausalLM` | `hy_v3` | `HYV3ForCausalLM` |
| `HCXVisionForCausalLM` | `hyperclovax_vision` | `HCXVisionForCausalLM` |
| `HCXVisionV2ForCausalLM` | `hyperclovax_vision_v2` | `HCXVisionV2ForCausalLM` |
| `HyperCLOVAXForCausalLM` | `hyperclovax` | `HyperCLOVAXForCausalLM` |
| `InternLMForCausalLM` | `llama` | `LlamaForCausalLM` |
| `InternLM2ForCausalLM` | `internlm2` | `InternLM2ForCausalLM` |
| `InternLM2VEForCausalLM` | `internlm2_ve` | `InternLM2VEForCausalLM` |
| `InternLM3ForCausalLM` | `llama` | `LlamaForCausalLM` |
| `IQuestCoderForCausalLM` | `llama` | `LlamaForCausalLM` |
| `IQuestLoopCoderForCausalLM` | `iquest_loopcoder` | `IQuestLoopCoderForCausalLM` |
| `JAISLMHeadModel` | `jais` | `JAISLMHeadModel` |
| `Jais2ForCausalLM` | `jais2` | `Jais2ForCausalLM` |
| `JambaForCausalLM` | `jamba` | `JambaForCausalLM` |
| `KimiLinearForCausalLM` | `kimi_linear` | `KimiLinearForCausalLM` |
| `Lfm2ForCausalLM` | `lfm2` | `Lfm2ForCausalLM` |
| `Lfm2MoeForCausalLM` | `lfm2_moe` | `Lfm2MoeForCausalLM` |
| `LagunaForCausalLM` | `laguna` | `LagunaForCausalLM` |
| `LlamaForCausalLM` | `llama` | `LlamaForCausalLM` |
| `Llama4ForCausalLM` | `llama4` | `Llama4ForCausalLM` |
| `LLaMAForCausalLM` | `llama` | `LlamaForCausalLM` |
| `LongcatFlashForCausalLM` | `longcat_flash` | `LongcatFlashForCausalLM` |
| `MambaForCausalLM` | `mamba` | `MambaForCausalLM` |
| `Mamba2ForCausalLM` | `mamba2` | `Mamba2ForCausalLM` |
| `MiniCPMForCausalLM` | `minicpm` | `MiniCPMForCausalLM` |
| `MiniCPM3ForCausalLM` | `minicpm3` | `MiniCPM3ForCausalLM` |
| `MiniMaxForCausalLM` | `minimax_text_01` | `MiniMaxText01ForCausalLM` |
| `MiniMaxText01ForCausalLM` | `minimax_text_01` | `MiniMaxText01ForCausalLM` |
| `MiniMaxM1ForCausalLM` | `minimax_text_01` | `MiniMaxText01ForCausalLM` |
| `MiniMaxM2ForCausalLM` | `minimax_m2` | `MiniMaxM2ForCausalLM` |
| `Ministral3ForCausalLM` | `mistral` | `MistralForCausalLM` |
| `MistralForCausalLM` | `mistral` | `MistralForCausalLM` |
| `MistralLarge3ForCausalLM` | `mistral_large_3` | `MistralLarge3ForCausalLM` |
| `MixtralForCausalLM` | `mixtral` | `MixtralForCausalLM` |
| `MptForCausalLM` | `mpt` | `MPTForCausalLM` |
| `MPTForCausalLM` | `mpt` | `MPTForCausalLM` |
| `MiMoForCausalLM` | `mimo` | `MiMoForCausalLM` |
| `MiMoV2FlashForCausalLM` | `mimo_v2` | `MiMoV2FlashForCausalLM` |
| `MiMoV2ForCausalLM` | `mimo_v2` | `MiMoV2ForCausalLM` |
| `NemotronForCausalLM` | `nemotron` | `NemotronForCausalLM` |
| `NemotronHForCausalLM` | `nemotron_h` | `NemotronHForCausalLM` |
| `NemotronHPuzzleForCausalLM` | `nemotron_h` | `NemotronHForCausalLM` |
| `OlmoForCausalLM` | `olmo` | `OlmoForCausalLM` |
| `Olmo2ForCausalLM` | `olmo2` | `Olmo2ForCausalLM` |
| `Olmo3ForCausalLM` | `olmo2` | `Olmo2ForCausalLM` |
| `OlmoHybridForCausalLM` | `olmo_hybrid` | `OlmoHybridForCausalLM` |
| `OlmoeForCausalLM` | `olmoe` | `OlmoeForCausalLM` |
| `OPTForCausalLM` | `opt` | `OPTForCausalLM` |
| `OrionForCausalLM` | `orion` | `OrionForCausalLM` |
| `OuroForCausalLM` | `ouro` | `OuroForCausalLM` |
| `PanguEmbeddedForCausalLM` | `openpangu` | `PanguEmbeddedForCausalLM` |
| `PanguProMoEV2ForCausalLM` | `openpangu` | `PanguProMoEV2ForCausalLM` |
| `PanguUltraMoEForCausalLM` | `openpangu` | `PanguUltraMoEForCausalLM` |
| `Param2MoEForCausalLM` | `param2moe` | `Param2MoEForCausalLM` |
| `PersimmonForCausalLM` | `persimmon` | `PersimmonForCausalLM` |
| `PhiForCausalLM` | `phi` | `PhiForCausalLM` |
| `Phi3ForCausalLM` | `phi3` | `Phi3ForCausalLM` |
| `PhiMoEForCausalLM` | `phimoe` | `PhiMoEForCausalLM` |
| `Plamo2ForCausalLM` | `plamo2` | `Plamo2ForCausalLM` |
| `Plamo3ForCausalLM` | `plamo3` | `Plamo3ForCausalLM` |
| `QWenLMHeadModel` | `qwen` | `QWenLMHeadModel` |
| `Qwen2ForCausalLM` | `qwen2` | `Qwen2ForCausalLM` |
| `Qwen2MoeForCausalLM` | `qwen2_moe` | `Qwen2MoeForCausalLM` |
| `Qwen3ForCausalLM` | `qwen3` | `Qwen3ForCausalLM` |
| `Qwen3MoeForCausalLM` | `qwen3_moe` | `Qwen3MoeForCausalLM` |
| `RWForCausalLM` | `falcon` | `FalconForCausalLM` |
| `SarvamMoEForCausalLM` | `sarvam` | `SarvamMoEForCausalLM` |
| `SarvamMLAForCausalLM` | `sarvam` | `SarvamMLAForCausalLM` |
| `SeedOssForCausalLM` | `seed_oss` | `SeedOssForCausalLM` |
| `Step1ForCausalLM` | `step1` | `Step1ForCausalLM` |
| `Step3TextForCausalLM` | `step3_text` | `Step3TextForCausalLM` |
| `Step3p5ForCausalLM` | `step3p5` | `Step3p5ForCausalLM` |
| `StableLMEpochForCausalLM` | `stablelm` | `StablelmForCausalLM` |
| `StableLmForCausalLM` | `stablelm` | `StablelmForCausalLM` |
| `Starcoder2ForCausalLM` | `starcoder2` | `Starcoder2ForCausalLM` |
| `SolarForCausalLM` | `solar` | `SolarForCausalLM` |
| `TeleChatForCausalLM` | `telechat2` | `TeleChat2ForCausalLM` |
| `TeleChat2ForCausalLM` | `telechat2` | `TeleChat2ForCausalLM` |
| `TeleChat3ForCausalLM` | `llama` | `LlamaForCausalLM` |
| `TeleFLMForCausalLM` | `teleflm` | `TeleFLMForCausalLM` |
| `XverseForCausalLM` | `llama` | `LlamaForCausalLM` |
| `Zamba2ForCausalLM` | `zamba2` | `Zamba2ForCausalLM` |

### Architecture Aliases

Several architectures are mapped to shared implementations:

| Alias Architecture | Resolved To |
|-------------------|-------------|
| `AquilaModel`, `AquilaForCausalLM` | `llama.LlamaForCausalLM` |
| `CwmForCausalLM` | `llama.LlamaForCausalLM` |
| `FalconMambaForCausalLM` | `mamba.MambaForCausalLM` |
| `InternLMForCausalLM`, `InternLM3ForCausalLM` | `llama.LlamaForCausalLM` |
| `IQuestCoderForCausalLM` | `llama.LlamaForCausalLM` |
| `LLaMAForCausalLM` | `llama.LlamaForCausalLM` |
| `Ministral3ForCausalLM` | `mistral.MistralForCausalLM` |
| `RWForCausalLM` | `falcon.FalconForCausalLM` |
| `TeleChat3ForCausalLM` | `llama.LlamaForCausalLM` |
| `XverseForCausalLM` | `llama.LlamaForCausalLM` |
| `NemotronHPuzzleForCausalLM` | `nemotron_h.NemotronHForCausalLM` |
| `DeepseekV32ForCausalLM` | `deepseek_v2.DeepseekV3ForCausalLM` |
| `MiniMaxForCausalLM`, `MiniMaxM1ForCausalLM` | `minimax_text_01.MiniMaxText01ForCausalLM` |
| `Olmo3ForCausalLM` | `olmo2.Olmo2ForCausalLM` |

---

## Embedding Models

**Source:** `_EMBEDDING_MODELS` in `registry.py`

Models that produce embeddings (dense vectors) from input text or multimodal inputs.

### Text-Only Embedding Models

| Architecture | Module | Class |
|-------------|--------|-------|
| `BertModel` | `bert` | `BertEmbeddingModel` |
| `BertSpladeSparseEmbeddingModel` | `bert` | `BertSpladeSparseEmbeddingModel` |
| `ErnieModel` | `ernie` | `ErnieEmbeddingModel` |
| `BgeM3EmbeddingModel` | `roberta` | `BgeM3EmbeddingModel` |
| `Gemma2Model` | `gemma2` | `Gemma2ForCausalLM` |
| `Gemma3TextModel` | `gemma3` | `Gemma3Model` |
| `GlmForCausalLM` | `glm` | `GlmForCausalLM` |
| `GritLM` | `gritlm` | `GritLM` |
| `GteModel` | `bert_with_rope` | `SnowflakeGteNewModel` |
| `GteNewModel` | `bert_with_rope` | `GteNewModel` |
| `JinaEmbeddingsV5Model` | `jina` | `JinaEmbeddingsV5Model` |
| `LlamaBidirectionalModel` | `llama` | `LlamaBidirectionalModel` |
| `LlamaModel` | `llama` | `LlamaForCausalLM` |
| `MistralModel` | `llama` | `LlamaForCausalLM` |
| `ModernBertModel` | `modernbert` | `ModernBertModel` |
| `NomicBertModel` | `bert_with_rope` | `NomicBertModel` |
| `Phi3ForCausalLM` | `phi3` | `Phi3ForCausalLM` |
| `Qwen2Model` | `qwen2` | `Qwen2ForCausalLM` |
| `Qwen2ForCausalLM` | `qwen2` | `Qwen2ForCausalLM` |
| `RobertaForMaskedLM` | `roberta` | `RobertaEmbeddingModel` |
| `RobertaModel` | `roberta` | `RobertaEmbeddingModel` |
| `TeleChatForCausalLM` | `telechat2` | `TeleChat2ForCausalLM` |
| `TeleChat2ForCausalLM` | `telechat2` | `TeleChat2ForCausalLM` |
| `VoyageQwen3BidirectionalEmbedModel` | `voyage` | `VoyageQwen3BidirectionalEmbedModel` |
| `XLMRobertaModel` | `roberta` | `RobertaEmbeddingModel` |
| `DeciLMForCausalLM` | `nemotron_nas` | `DeciLMForCausalLM` |

Note: `_EMBEDDING_MODELS` also includes all models whose architecture resolves to `LlamaForCausalLM` (from `_TEXT_GENERATION_MODELS`), so they can be used for embedding tasks as well.

### Multimodal Embedding Models

| Architecture | Module | Class |
|-------------|--------|-------|
| `CLIPModel` | `clip` | `CLIPEmbeddingModel` |
| `ColPaliForRetrieval` | `colpali` | `ColPaliModel` |
| `LlamaNemotronVLModel` | `nemotron_vl` | `LlamaNemotronVLForEmbedding` |
| `LlavaNextForConditionalGeneration` | `llava_next` | `LlavaNextForConditionalGeneration` |
| `Phi3VForCausalLM` | `phi3v` | `Phi3VForCausalLM` |
| `Qwen2VLForConditionalGeneration` | `qwen2_vl` | `Qwen2VLForConditionalGeneration` |
| `SiglipModel` | `siglip` | `SiglipEmbeddingModel` |
| `PrithviGeoSpatialMAE` | `terratorch` | `Terratorch` |
| `Terratorch` | `terratorch` | `Terratorch` |

---

## Late Interaction Models

**Source:** `_LATE_INTERACTION_MODELS` in `registry.py`

Models for late interaction retrieval (e.g., ColBERT-style MaxSim scoring).

### Text-Only

| Architecture | Module | Class |
|-------------|--------|-------|
| `HF_ColBERT` | `colbert` | `ColBERTModel` |
| `ColBERTModernBertModel` | `colbert` | `ColBERTModernBertModel` |
| `ColBERTJinaRobertaModel` | `colbert` | `ColBERTJinaRobertaModel` |
| `ColBERTLfm2Model` | `colbert` | `ColBERTLfm2Model` |
| `JinaForRanking` | `jina` | `JinaForRanking` |

### Multimodal

| Architecture | Module | Class |
|-------------|--------|-------|
| `ColModernVBertForRetrieval` | `colmodernvbert` | `ColModernVBertForRetrieval` |
| `ColPaliForRetrieval` | `colpali` | `ColPaliModel` |
| `ColQwen3` | `colqwen3` | `ColQwen3Model` |
| `OpsColQwen3Model` | `colqwen3` | `ColQwen3Model` |
| `ColQwen3_5` | `colqwen3_5` | `ColQwen3_5Model` |
| `Qwen3VLNemotronEmbedModel` | `colqwen3` | `ColQwen3Model` |

### Late Interaction Utilities

**Source:** `vllm/v1/pool/late_interaction.py`

#### Constants

```python
LATE_INTERACTION_MODE_CACHE_QUERY = "cache_query"
LATE_INTERACTION_MODE_SCORE_DOC = "score_doc"
```

#### Functions

- `get_late_interaction_engine_index(pooling_params: PoolingParams | None, num_engines: int) -> int | None` - Determines which engine to pin a request to based on CRC32 of the query key. Returns `None` if no late interaction params.

- `build_late_interaction_query_params(query_key: str, query_uses: int) -> LateInteractionParams` - Builds params for the cache_query mode.

- `build_late_interaction_doc_params(query_key: str) -> LateInteractionParams` - Builds params for the score_doc mode.

- `compute_maxsim_score_batched(q_embs: Sequence[torch.Tensor], d_embs: Sequence[torch.Tensor], max_batch_size: int = 64, max_score_matrix_elements: int = 64_000_000) -> list[torch.Tensor]` - Computes MaxSim scores for multiple query-document pairs in mini-batches.

---

## Reward Models

**Source:** `_REWARD_MODELS` in `registry.py`

| Architecture | Module | Class |
|-------------|--------|-------|
| `InternLM2ForRewardModel` | `internlm2` | `InternLM2ForRewardModel` |
| `Qwen2ForRewardModel` | `qwen2_rm` | `Qwen2ForRewardModel` |
| `Qwen2ForProcessRewardModel` | `qwen2_rm` | `Qwen2ForProcessRewardModel` |

---

## Token Classification Models

**Source:** `_TOKEN_CLASSIFICATION_MODELS` in `registry.py`

| Architecture | Module | Class |
|-------------|--------|-------|
| `BertForTokenClassification` | `bert` | `BertForTokenClassification` |
| `ErnieForTokenClassification` | `ernie` | `ErnieForTokenClassification` |
| `ModernBertForTokenClassification` | `modernbert` | `ModernBertForTokenClassification` |
| `Qwen3ASRForcedAlignerForTokenClassification` | `qwen3_asr_forced_aligner` | `Qwen3ASRForcedAlignerForTokenClassification` |

---

## Sequence Classification Models

**Source:** `_SEQUENCE_CLASSIFICATION_MODELS` in `registry.py`

### Text-Only

| Architecture | Module | Class |
|-------------|--------|-------|
| `BertForSequenceClassification` | `bert` | `BertForSequenceClassification` |
| `GPT2ForSequenceClassification` | `gpt2` | `GPT2ForSequenceClassification` |
| `ErnieForSequenceClassification` | `ernie` | `ErnieForSequenceClassification` |
| `GteNewForSequenceClassification` | `bert_with_rope` | `GteNewForSequenceClassification` |
| `JambaForSequenceClassification` | `jamba` | `JambaForSequenceClassification` |
| `LlamaBidirectionalForSequenceClassification` | `llama` | `LlamaBidirectionalForSequenceClassification` |
| `ModernBertForSequenceClassification` | `modernbert` | `ModernBertForSequenceClassification` |
| `RobertaForSequenceClassification` | `roberta` | `RobertaForSequenceClassification` |
| `XLMRobertaForSequenceClassification` | `roberta` | `RobertaForSequenceClassification` |

### Multimodal

| Architecture | Module | Class |
|-------------|--------|-------|
| `JinaVLForRanking` | `jina_vl` | `JinaVLForSequenceClassification` |
| `LlamaNemotronVLForSequenceClassification` | `nemotron_vl` | `LlamaNemotronVLForSequenceClassification` |

---

## Multimodal Models

**Source:** `_MULTIMODAL_MODELS` in `registry.py`

Models that accept and process multimodal inputs (images, audio, video).

### Complete Catalog

| Architecture | Module | Class |
|-------------|--------|-------|
| `AriaForConditionalGeneration` | `aria` | `AriaForConditionalGeneration` |
| `AudioFlamingo3ForConditionalGeneration` | `audioflamingo3` | `AudioFlamingo3ForConditionalGeneration` |
| `MusicFlamingoForConditionalGeneration` | `musicflamingo` | `MusicFlamingoForConditionalGeneration` |
| `AyaVisionForConditionalGeneration` | `aya_vision` | `AyaVisionForConditionalGeneration` |
| `BagelForConditionalGeneration` | `bagel` | `BagelForConditionalGeneration` |
| `BeeForConditionalGeneration` | `bee` | `BeeForConditionalGeneration` |
| `Blip2ForConditionalGeneration` | `blip2` | `Blip2ForConditionalGeneration` |
| `ChameleonForConditionalGeneration` | `chameleon` | `ChameleonForConditionalGeneration` |
| `Cheers` | `cheers` | `CheersForConditionalGeneration` |
| `CheersForConditionalGeneration` | `cheers` | `CheersForConditionalGeneration` |
| `Cohere2VisionForConditionalGeneration` | `cohere2_vision` | `Cohere2VisionForConditionalGeneration` |
| `DeepseekVLV2ForCausalLM` | `deepseek_vl2` | `DeepseekVLV2ForCausalLM` |
| `DeepseekOCRForCausalLM` | `deepseek_ocr` | `DeepseekOCRForCausalLM` |
| `DeepseekOCR2ForCausalLM` | `deepseek_ocr2` | `DeepseekOCR2ForCausalLM` |
| `DotsOCRForCausalLM` | `dots_ocr` | `DotsOCRForCausalLM` |
| `Eagle2_5_VLForConditionalGeneration` | `eagle2_5_vl` | `Eagle2_5_VLForConditionalGeneration` |
| `Ernie4_5_VLMoeForConditionalGeneration` | `ernie45_vl` | `Ernie4_5_VLMoeForConditionalGeneration` |
| `Exaone4_5_ForConditionalGeneration` | `exaone4_5` | `Exaone4_5_ForConditionalGeneration` |
| `FireRedASR2ForConditionalGeneration` | `fireredasr2` | `FireRedASR2ForConditionalGeneration` |
| `FunASRForConditionalGeneration` | `funasr` | `FunASRForConditionalGeneration` |
| `FireRedLIDForConditionalGeneration` | `fireredlid` | `FireRedLIDForConditionalGeneration` |
| `FunAudioChatForConditionalGeneration` | `funaudiochat` | `FunAudioChatForConditionalGeneration` |
| `FuyuForCausalLM` | `fuyu` | `FuyuForCausalLM` |
| `Gemma3ForConditionalGeneration` | `gemma3_mm` | `Gemma3ForConditionalGeneration` |
| `Gemma3nForConditionalGeneration` | `gemma3n_mm` | `Gemma3nForConditionalGeneration` |
| `Gemma4ForConditionalGeneration` | `gemma4_mm` | `Gemma4ForConditionalGeneration` |
| `GlmAsrForConditionalGeneration` | `glmasr` | `GlmAsrForConditionalGeneration` |
| `GLM4VForCausalLM` | `glm4v` | `GLM4VForCausalLM` |
| `Glm4vForConditionalGeneration` | `glm4_1v` | `Glm4vForConditionalGeneration` |
| `Glm4vMoeForConditionalGeneration` | `glm4_1v` | `Glm4vMoeForConditionalGeneration` |
| `GlmOcrForConditionalGeneration` | `glm_ocr` | `GlmOcrForConditionalGeneration` |
| `GraniteSpeechForConditionalGeneration` | `granite_speech` | `GraniteSpeechForConditionalGeneration` |
| `Granite4VisionForConditionalGeneration` | `granite4_vision` | `Granite4VisionForConditionalGeneration` |
| `H2OVLChatModel` | `h2ovl` | `H2OVLChatModel` |
| `HunYuanVLForConditionalGeneration` | `hunyuan_vision` | `HunYuanVLForConditionalGeneration` |
| `InternVLChatModel` | `internvl` | `InternVLChatModel` |
| `InternS1ForConditionalGeneration` | `interns1` | `InternS1ForConditionalGeneration` |
| `InternVLForConditionalGeneration` | `interns1` | `InternS1ForConditionalGeneration` |
| `InternS1ProForConditionalGeneration` | `interns1_pro` | `InternS1ProForConditionalGeneration` |
| `Idefics3ForConditionalGeneration` | `idefics3` | `Idefics3ForConditionalGeneration` |
| `IsaacForConditionalGeneration` | `isaac` | `IsaacForConditionalGeneration` |
| `KananaVForConditionalGeneration` | `kanana_v` | `KananaVForConditionalGeneration` |
| `KeyeForConditionalGeneration` | `keye` | `KeyeForConditionalGeneration` |
| `KeyeVL1_5ForConditionalGeneration` | `keye_vl1_5` | `KeyeVL1_5ForConditionalGeneration` |
| `KimiVLForConditionalGeneration` | `kimi_vl` | `KimiVLForConditionalGeneration` |
| `KimiK25ForConditionalGeneration` | `kimi_k25` | `KimiK25ForConditionalGeneration` |
| `MoonshotKimiaForCausalLM` | `kimi_audio` | `KimiAudioForConditionalGeneration` |
| `LightOnOCRForConditionalGeneration` | `lightonocr` | `LightOnOCRForConditionalGeneration` |
| `Lfm2VlForConditionalGeneration` | `lfm2_vl` | `Lfm2VLForConditionalGeneration` |
| `Llama4ForConditionalGeneration` | `mllama4` | `Llama4ForConditionalGeneration` |
| `Llama_Nemotron_Nano_VL` | `nemotron_vl` | `LlamaNemotronVLChatModel` |
| `LlavaForConditionalGeneration` | `llava` | `LlavaForConditionalGeneration` |
| `LlavaNextForConditionalGeneration` | `llava_next` | `LlavaNextForConditionalGeneration` |
| `LlavaNextVideoForConditionalGeneration` | `llava_next_video` | `LlavaNextVideoForConditionalGeneration` |
| `LlavaOnevisionForConditionalGeneration` | `llava_onevision` | `LlavaOnevisionForConditionalGeneration` |
| `MantisForConditionalGeneration` | `llava` | `MantisForConditionalGeneration` |
| `MiDashengLMModel` | `midashenglm` | `MiDashengLMModel` |
| `MiMoV2OmniForCausalLM` | `mimo_v2_omni` | `MiMoV2OmniForCausalLM` |
| `MiniMaxVL01ForConditionalGeneration` | `minimax_vl_01` | `MiniMaxVL01ForConditionalGeneration` |
| `MiniCPMO` | `minicpmo` | `MiniCPMO` |
| `MiniCPMV` | `minicpmv` | `MiniCPMV` |
| `Mistral3ForConditionalGeneration` | `mistral3` | `Mistral3ForConditionalGeneration` |
| `MolmoForCausalLM` | `molmo` | `MolmoForCausalLM` |
| `Molmo2ForConditionalGeneration` | `molmo2` | `Molmo2ForConditionalGeneration` |
| `Moondream3ForCausalLM` | `moondream3` | `Moondream3ForCausalLM` |
| `HfMoondream` | `moondream3` | `Moondream3ForCausalLM` |
| `NemotronH_Nano_VL_V2` | `nano_nemotron_vl` | `NemotronH_Nano_VL_V2` |
| `NemotronH_Nano_Omni_Reasoning_V3` | `nano_nemotron_vl` | `NemotronH_Nano_VL_V2` |
| `NemotronH_Super_Omni_Reasoning_V3` | `nano_nemotron_vl` | `NemotronH_Nano_VL_V2` |
| `NVLM_D` | `nvlm_d` | `NVLM_D_Model` |
| `OpenCUAForConditionalGeneration` | `opencua` | `OpenCUAForConditionalGeneration` |
| `OpenPanguVLForConditionalGeneration` | `openpangu_vl` | `OpenPanguVLForConditionalGeneration` |
| `Ovis` | `ovis` | `Ovis` |
| `Ovis2_5` | `ovis2_5` | `Ovis2_5` |
| `Ovis2_6ForCausalLM` | `ovis2_5` | `Ovis2_5` |
| `Ovis2_6_MoeForCausalLM` | `ovis2_5` | `Ovis2_5` |
| `PaddleOCRVLForConditionalGeneration` | `paddleocr_vl` | `PaddleOCRVLForConditionalGeneration` |
| `PaliGemmaForConditionalGeneration` | `paligemma` | `PaliGemmaForConditionalGeneration` |
| `Phi3VForCausalLM` | `phi3v` | `Phi3VForCausalLM` |
| `Phi4ForCausalLMV` | `phi4siglip` | `Phi4ForCausalLMV` |
| `Phi4MMForCausalLM` | `phi4mm` | `Phi4MMForCausalLM` |
| `PixtralForConditionalGeneration` | `pixtral` | `PixtralForConditionalGeneration` |
| `QianfanOCRForConditionalGeneration` | `qianfan_ocr` | `QianfanOCRForConditionalGeneration` |
| `QwenVLForConditionalGeneration` | `qwen_vl` | `QwenVLForConditionalGeneration` |
| `Qwen2VLForConditionalGeneration` | `qwen2_vl` | `Qwen2VLForConditionalGeneration` |
| `Qwen2_5_VLForConditionalGeneration` | `qwen2_5_vl` | `Qwen2_5_VLForConditionalGeneration` |
| `Qwen2AudioForConditionalGeneration` | `qwen2_audio` | `Qwen2AudioForConditionalGeneration` |
| `Qwen2_5OmniModel` | `qwen2_5_omni_thinker` | `Qwen2_5OmniThinkerForConditionalGeneration` |
| `Qwen2_5OmniForConditionalGeneration` | `qwen2_5_omni_thinker` | `Qwen2_5OmniThinkerForConditionalGeneration` |
| `Qwen3OmniMoeForConditionalGeneration` | `qwen3_omni_moe_thinker` | `Qwen3OmniMoeThinkerForConditionalGeneration` |
| `Qwen3ASRForConditionalGeneration` | `qwen3_asr` | `Qwen3ASRForConditionalGeneration` |
| `Qwen3ASRRealtimeGeneration` | `qwen3_asr_realtime` | `Qwen3ASRRealtimeGeneration` |
| `Qwen3VLForConditionalGeneration` | `qwen3_vl` | `Qwen3VLForConditionalGeneration` |
| `Qwen3VLMoeForConditionalGeneration` | `qwen3_vl_moe` | `Qwen3VLMoeForConditionalGeneration` |
| `Qwen3_5ForConditionalGeneration` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` |
| `Qwen3_5MoeForConditionalGeneration` | `qwen3_5` | `Qwen3_5MoeForConditionalGeneration` |
| `RForConditionalGeneration` | `rvl` | `RForConditionalGeneration` |
| `SkyworkR1VChatModel` | `skyworkr1v` | `SkyworkR1VChatModel` |
| `SmolVLMForConditionalGeneration` | `smolvlm` | `SmolVLMForConditionalGeneration` |
| `StepVLForConditionalGeneration` | `step_vl` | `StepVLForConditionalGeneration` |
| `Step3VLForConditionalGeneration` | `step3_vl` | `Step3VLForConditionalGeneration` |
| `TarsierForConditionalGeneration` | `tarsier` | `TarsierForConditionalGeneration` |
| `Tarsier2ForConditionalGeneration` | `qwen2_vl` | `Tarsier2ForConditionalGeneration` |
| `UltravoxModel` | `ultravox` | `UltravoxModel` |
| `VoxtralForConditionalGeneration` | `voxtral` | `VoxtralForConditionalGeneration` |
| `VoxtralRealtimeGeneration` | `voxtral_realtime` | `VoxtralRealtimeGeneration` |

### Encoder-Decoder Multimodal

| Architecture | Module | Class |
|-------------|--------|-------|
| `CohereAsrForConditionalGeneration` | `cohere_asr` | `CohereAsrForConditionalGeneration` |
| `NemotronParseForConditionalGeneration` | `nemotron_parse` | `NemotronParseForConditionalGeneration` |
| `WhisperForConditionalGeneration` | `whisper` | `WhisperForConditionalGeneration` |

---

## Speculative Decoding Models

**Source:** `_SPECULATIVE_DECODING_MODELS` in `registry.py`

Models used as draft models for speculative decoding.

| Architecture | Module | Class |
|-------------|--------|-------|
| `ExtractHiddenStatesModel` | `extract_hidden_states` | `ExtractHiddenStatesModel` |
| `MiMoMTPModel` | `mimo_mtp` | `MiMoMTP` |
| `MiMoV2MTPModel` | `mimo_v2_mtp` | `MiMoV2MTP` |
| `MiMoV2OmniMTPModel` | `mimo_v2_mtp` | `MiMoV2OmniMTP` |
| `EagleLlamaForCausalLM` | `llama_eagle` | `EagleLlamaForCausalLM` |
| `EagleLlama4ForCausalLM` | `llama4_eagle` | `EagleLlama4ForCausalLM` |
| `EagleMiniCPMForCausalLM` | `minicpm_eagle` | `EagleMiniCPMForCausalLM` |
| `DFlashDraftModel` | `qwen3_dflash` | `DFlashQwen3ForCausalLM` |
| `Eagle3LlamaForCausalLM` | `llama_eagle3` | `Eagle3LlamaForCausalLM` |
| `Eagle3MiniMaxM2ForCausalLM` | `llama_eagle3` | `Eagle3LlamaForCausalLM` |
| `LlamaForCausalLMEagle3` | `llama_eagle3` | `Eagle3LlamaForCausalLM` |
| `Eagle3Qwen2_5vlForCausalLM` | `llama_eagle3` | `Eagle3LlamaForCausalLM` |
| `Eagle3Qwen3vlForCausalLM` | `llama_eagle3` | `Eagle3LlamaForCausalLM` |
| `EagleMistralForCausalLM` | `mistral_eagle` | `EagleMistralForCausalLM` |
| `EagleMistralLarge3ForCausalLM` | `mistral_large_3_eagle` | `EagleMistralLarge3ForCausalLM` |
| `Eagle3DeepseekV2ForCausalLM` | `deepseek_eagle3` | `Eagle3DeepseekV2ForCausalLM` |
| `Eagle3DeepseekV3ForCausalLM` | `deepseek_eagle3` | `Eagle3DeepseekV2ForCausalLM` |
| `EagleDeepSeekMTPModel` | `deepseek_eagle` | `EagleDeepseekV3ForCausalLM` |
| `DeepSeekMTPModel` | `deepseek_mtp` | `DeepSeekMTP` |
| `DeepSeekV4MTPModel` | `deepseek_v4_mtp` | `DeepSeekV4MTP` |
| `ErnieMTPModel` | `ernie_mtp` | `ErnieMTP` |
| `ExaoneMoeMTP` | `exaone_moe_mtp` | `ExaoneMoeMTP` |
| `Exaone4_5_MTP` | `exaone4_5_mtp` | `Exaone4_5_MTP` |
| `NemotronHMTPModel` | `nemotron_h_mtp` | `NemotronHMTP` |
| `LongCatFlashMTPModel` | `longcat_flash_mtp` | `LongCatFlashMTP` |
| `Glm4MoeMTPModel` | `glm4_moe_mtp` | `Glm4MoeMTP` |
| `Glm4MoeLiteMTPModel` | `glm4_moe_lite_mtp` | `Glm4MoeLiteMTP` |
| `GlmOcrMTPModel` | `glm_ocr_mtp` | `GlmOcrMTP` |
| `MedusaModel` | `medusa` | `Medusa` |
| `OpenPanguMTPModel` | `openpangu_mtp` | `OpenPanguMTP` |
| `Qwen3NextMTP` | `qwen3_next_mtp` | `Qwen3NextMTP` |
| `Step3p5MTP` | `step3p5_mtp` | `Step3p5MTP` |
| `Qwen3_5MTP` | `qwen3_5_mtp` | `Qwen3_5MTP` |
| `Qwen3_5MoeMTP` | `qwen3_5_mtp` | `Qwen3_5MoeMTP` |
| `HYV3MTPModel` | `hy_v3_mtp` | `HYV3MTP` |

Note: `MLPSpeculatorPreTrainedModel` is temporarily disabled.

---

## Transformers Backend Models

**Source:** `_TRANSFORMERS_SUPPORTED_MODELS` and `_TRANSFORMERS_BACKEND_MODELS`

Models that can run through the generic Transformers backend.

### Directly Supported (from HuggingFace configs)

| Architecture | Module | Class |
|-------------|--------|-------|
| `SmolLM3ForCausalLM` | `transformers` | `TransformersForCausalLM` |
| `Emu3ForConditionalGeneration` | `transformers` | `TransformersMultiModalForCausalLM` |

### Backend Implementation Classes

| Architecture | Module | Class | Purpose |
|-------------|--------|-------|---------|
| `TransformersForCausalLM` | `transformers` | `TransformersForCausalLM` | Dense text generation |
| `TransformersMoEForCausalLM` | `transformers` | `TransformersMoEForCausalLM` | MoE text generation |
| `TransformersMultiModalForCausalLM` | `transformers` | `TransformersMultiModalForCausalLM` | Dense multimodal generation |
| `TransformersMultiModalMoEForCausalLM` | `transformers` | `TransformersMultiModalMoEForCausalLM` | MoE multimodal generation |
| `TransformersEmbeddingModel` | `transformers` | `TransformersEmbeddingModel` | Dense embeddings |
| `TransformersMoEEmbeddingModel` | `transformers` | `TransformersMoEEmbeddingModel` | MoE embeddings |
| `TransformersMultiModalEmbeddingModel` | `transformers` | `TransformersMultiModalEmbeddingModel` | Multimodal embeddings |
| `TransformersForSequenceClassification` | `transformers` | `TransformersForSequenceClassification` | Dense sequence classification |
| `TransformersMoEForSequenceClassification` | `transformers` | `TransformersMoEForSequenceClassification` | MoE sequence classification |
| `TransformersMultiModalForSequenceClassification` | `transformers` | `TransformersMultiModalForSequenceClassification` | Multimodal sequence classification |

---

## Previously Supported Models

**Source:** `_PREVIOUSLY_SUPPORTED_MODELS` in `registry.py`

Models that were previously supported but have been removed. The error message directs users to an older vLLM version.

| Architecture | Last Supported Version |
|-------------|----------------------|
| `MotifForCausalLM` | v0.10.2 |
| `Phi3SmallForCausalLM` | v0.9.2 |
| `Phi4FlashForCausalLM` | v0.10.2 |
| `Phi4MultimodalForCausalLM` | v0.12.0 |
| `DonutForConditionalGeneration` | v0.10.2 |
| `MllamaForConditionalGeneration` | v0.10.2 |

---

## Out-of-Tree Plugin Models

**Source:** `_OOT_SUPPORTED_MODELS` in `registry.py`

Models that are no longer in-tree but can be added via external plugins.

| Architecture | Plugin URL |
|-------------|-----------|
| `BartModel` | https://github.com/vllm-project/bart-plugin |
| `BartForConditionalGeneration` | https://github.com/vllm-project/bart-plugin |
| `Florence2ForConditionalGeneration` | https://github.com/vllm-project/bart-plugin |
| `MBartForConditionalGeneration` | https://github.com/vllm-project/bart-plugin |

---

## Model Configuration (ModelConfig)

**Source:** `vllm/config/model.py`

The `ModelConfig` class contains all configuration for a specific model instance.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | required | Name or path of the HuggingFace model |
| `task` | `str` | `"auto"` | Task for the model: `generate`, `embedding`, `classify`, `reward`, `score`, `generate_embeds` |
| `tokenizer` | `str` | model name | Name or path of the tokenizer |
| `tokenizer_mode` | `str` | `"auto"` | Tokenizer mode: `auto`, `slow`, `mistral`, `custom` |
| `trust_remote_code` | `bool` | `False` | Trust remote code from HuggingFace |
| `dtype` | `str` | `"auto"` | Data type: `auto`, `float16`, `bfloat16`, `float32` |
| `max_model_len` | `int \| None` | `None` | Maximum sequence length |
| `quantization` | `str \| None` | `None` | Quantization method |
| `revision` | `str \| None` | `None` | Model revision |
| `code_revision` | `str \| None` | `None` | Code revision |
| `tokenizer_revision` | `str \| None` | `None` | Tokenizer revision |
| `model_impl` | `str` | `"auto"` | Implementation: `auto`, `vllm`, `transformers`, `terratorch` |
| `runner_type` | `str` | `"auto"` | Runner: `auto`, `generate`, `pooling`, `encode`, `transcribe` |
| `convert_type` | `str` | `"auto"` | Convert: `auto`, `none`, `embedding`, `classify`, `reward`, `score` |
| `seed` | `int` | `0` | Random seed |
| `hf_overrides` | `dict \| None` | `None` | Overrides for HuggingFace config |
| `enforce_eager` | `bool` | `False` | Always use eager mode (no CUDA graphs) |
| `max_seq_len_to_capture` | `int` | `8192` | Maximum sequence length for CUDA graph capture |
| `max_logprobs` | `int` | `20` | Maximum number of logprobs |
| `hf_config` | `PretrainedConfig` | auto | HuggingFace config |
| `override_pooler_config` | `PoolerConfig \| None` | `None` | Pooler configuration override |
| `logits_processor_pattern` | `str \| None` | `None` | Regex for logits processor selection |
| `served_model_name` | `str \| list[str] \| None` | `None` | Model name(s) exposed in the API |
| `limit_mm_per_prompt` | `dict \| None` | `None` | Limits per multimodal modality per prompt |
| `media_io_kwargs` | `dict \| None` | `None` | Media I/O kwargs |
| `config_format` | `str` | `"auto"` | Config format: `auto`, `hf`, `mistral` |
| `mm_processor_kwargs` | `dict \| None` | `None` | Multimodal processor kwargs |
| `disable_mm_preprocessor_cache` | `bool` | `False` | Disable multimodal preprocessor cache |
| `override_multimodal_config` | `MultiModalConfig \| None` | `None` | Override multimodal config |
| `use_tqdm_on_load` | `bool` | `True` | Show progress bar during model loading |

### Key Methods

- `get_and_verify_max_len()` - Determines and validates the maximum model length from HF config.
- `verify_with_parallel_config(parallel_config)` - Verifies model config is compatible with parallel settings.
- `get_sliding_window()` - Returns sliding window size if applicable.
- `get_hidden_size()` - Returns the hidden dimension size.
- `get_total_num_kv_heads()` - Returns total number of KV heads.
- `get_head_size()` - Returns size of each attention head.
- `get_num_attention_heads()` - Returns number of attention heads.
- `get_vocab_size()` - Returns vocabulary size.
- `get_max_model_len()` - Returns maximum model length.
- `get_dtype()` - Returns resolved data type.

---

## Model Architecture Config

**Source:** `vllm/config/model_arch.py`

```python
@dataclass(frozen=True)
class ModelArchitectureConfig:
    architectures: list[str]
    model_type: str | None
    text_model_type: str | None
    hidden_size: int
    total_num_hidden_layers: int
    total_num_attention_heads: int
    head_size: int
    vocab_size: int
    total_num_kv_heads: int
    num_experts: int | None
    quantization_config: dict[str, Any] | None
    is_deepseek_mla: bool
    is_mm_prefix_lm: bool
    derived_max_model_len_and_key: tuple[int | None, str]
```

---

## Model-Specific Configuration Handlers

**Source:** `vllm/model_executor/models/config.py`

vLLM has a system of model-specific configuration handlers that adjust HuggingFace configs to work correctly with vLLM. Each handler is a subclass of `VerifyAndUpdateConfig` and registered in `MODELS_CONFIG_MAP`.

### Registered Handlers

| Key | Handler Class | Description |
|-----|--------------|-------------|
| `DeepseekV32` | `_DeepseekV32ConfigUpdater` | DeepSeek V3.2 config updates |
| `Ernie4_5_VL` | `_Ernie45VLConfigUpdater` | ERNIE 4.5 VL config updates |
| `Gemma3` | `_Gemma3ConfigUpdater` | Gemma3 config updates |
| `Gemma4` | `_Gemma4ConfigUpdater` | Gemma4 config updates |
| `DeepseekV4` | `_DeepseekV4ConfigUpdater` | DeepSeek V4 config updates |
| `GptOss` | `_GptOssConfigUpdater` | GPT-OSS config updates |
| `GteNew` | `_GteNewConfigUpdater` | GTE New embedding model updates |
| `HybridAttentionMamba` | `_HybridAttentionMambaConfigUpdater` | Hybrid attention/Mamba models |
| `Jamba` | `_JambaConfigUpdater` | Jamba config updates |
| `Jina` | `_JinaConfigUpdater` | Jina config updates |
| `LlamaBidirectional` | `_LlamaBidirectionalConfigUpdater` | Bidirectional Llama models |
| `LlamaNemotronVL` | `_LlamaNemotronVLConfigUpdater` | Llama Nemotron VL models |
| `Mamba` | `_MambaConfigUpdater` | Mamba models |
| `NemotronH` | `_NemotronHConfigUpdater` | Nemotron H models |
| `NomicBert` | `_NomicBertConfigUpdater` | NomicBERT models |
| `Qwen2RewardModel` | `_Qwen2RewardModelConfigUpdater` | Qwen2 reward model updates |
| `Qwen2ForSequenceClassification` | `_Qwen2SeqClsConfigUpdater` | Qwen2 sequence classification updates |
| `Qwen3_5` | `_Qwen3_5ConfigUpdater` | Qwen3.5 config updates |
| `SnowflakeGte` | `_SnowflakeGteConfigUpdater` | Snowflake GTE updates |
| `VoyageQwen3` | `_VoyageQwen3ConfigUpdater` | Voyage Qwen3 updates |

Each handler's `__call__` method takes `(config: PretrainedConfig) -> PretrainedConfig` and modifies it in place as needed.

---

## Model Capability Interfaces

**Source:** `vllm/model_executor/models/interfaces.py` and `interfaces_base.py`

Models declare their capabilities through interface mixins and decorators.

### Interface Functions (decorators/checkers)

| Function | Description |
|----------|-------------|
| `supports_multimodal(model)` | Returns True if the model supports multimodal inputs |
| `supports_pp(model)` | Returns True if the model supports pipeline parallelism |
| `supports_multimodal_encoder_tp_data(model)` | Returns True if encoder supports TP data |
| `supports_multimodal_raw_input_only(model)` | Returns True if model only accepts raw multimodal inputs |
| `requires_raw_input_tokens(model)` | Returns True if model requires raw token inputs |
| `has_inner_state(model)` | Returns True if model has internal state (e.g., KV cache) |
| `is_attention_free(model)` | Returns True if model has no attention layers (e.g., Mamba) |
| `is_hybrid(model)` | Returns True if model is hybrid (attention + state-space) |
| `has_noops(model)` | Returns True if model has no-op layers |
| `supports_mamba_prefix_caching(model)` | Returns True if Mamba model supports prefix caching |
| `supports_transcription(model)` | Returns True if model supports transcription (ASR) |

### Base Interface Functions

| Function | Description |
|----------|-------------|
| `is_text_generation_model(model)` | Returns True if model generates text |
| `is_pooling_model(model)` | Returns True if model produces pooled outputs |
| `get_attn_type(model)` | Returns attention type string |
| `get_default_seq_pooling_type(model)` | Returns default sequence pooling type |
| `get_default_tok_pooling_type(model)` | Returns default token pooling type |
| `get_score_type(model)` | Returns score type for ranking models |

### Capability Flags in _ModelInfo

The `_ModelInfo` dataclass captures all these capabilities:

```python
@dataclass(frozen=True)
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool
    is_pooling_model: bool
    attn_type: AttnTypeStr
    default_seq_pooling_type: SequencePoolingType
    default_tok_pooling_type: TokenPoolingType
    score_type: ScoreType
    supports_multimodal: bool
    supports_multimodal_raw_input_only: bool
    requires_raw_input_tokens: bool
    supports_multimodal_encoder_tp_data: bool
    supports_pp: bool
    has_inner_state: bool
    is_attention_free: bool
    is_hybrid: bool
    has_noops: bool
    supports_mamba_prefix_caching: bool
    supports_transcription: bool
    supports_transcription_only: bool
```

---

## Model Info Dataclass

The `_ModelInfo` dataclass is the core mechanism for querying model capabilities without loading the full model class. It is populated by inspecting the model class in a subprocess and cached to disk for subsequent lookups.

### Caching

Model info is cached as JSON files under `$VLLM_CACHE_ROOT/modelinfos/` with filenames like `<module_name>-<class_name>.json`. Each cache entry contains:

```json
{
  "hash": "<sha256 of module source file>",
  "modelinfo": {
    "architecture": "...",
    "is_text_generation_model": true,
    "is_pooling_model": false,
    ...
  }
}
```

The cache is invalidated when the source file's content hash changes.

### Inspection Flow

1. Check if the model source file has a cached `_ModelInfo` (hash-matched)
2. If cache hit, return cached `_ModelInfo`
3. If cache miss, run `_ModelInfo.from_model_cls()` in a subprocess
4. Save result to cache for future use

---

## Pooling Metadata

**Source:** `vllm/v1/pool/metadata.py`

### PoolingCursor

```python
@dataclass
class PoolingCursor:
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor
```

Tracks the position of first and last tokens for each sequence in a batch. Supports slicing via `__getitem__`.

Methods:
- `is_partial_prefill() -> bool` - Returns True if not all prompt tokens have been scheduled
- `is_finished() -> bool` - Returns True if all prompt tokens equal sequence length

### PoolingStates

```python
class PoolingStates:
    hidden_states_cache: list[torch.Tensor]
```

Simple container for caching hidden states during chunked prefill.

### PoolingMetadata

```python
@dataclass
class PoolingMetadata:
    prompt_lens: torch.Tensor           # CPU Tensor
    prompt_token_ids: torch.Tensor | None  # Model-device tensor
    prompt_token_ids_cpu: torch.Tensor | None  # CPU tensor
    pooling_params: list[PoolingParams]
    pooling_states: list[PoolingStates]
    pooling_cursor: PoolingCursor | None = None
```

Key methods:
- `get_prompt_token_ids() -> list[torch.Tensor]` - Returns per-prompt token ID tensors
- `get_prompt_token_ids_cpu() -> list[torch.Tensor]` - CPU variant
- `get_pooling_cursor() -> PoolingCursor` - Returns the cursor (must call `build_pooling_cursor` first)
- `build_pooling_cursor(num_scheduled_tokens_np, seq_lens_cpu, device, query_start_loc_gpu=None)` - Constructs the pooling cursor

---

## Serialization (Serial Utils)

**Source:** `vllm/v1/serial_utils.py`

vLLM uses custom msgpack-based serialization for inter-process communication (e.g., between API server and worker processes).

### MsgpackEncoder

```python
class MsgpackEncoder:
    def __init__(
        self,
        size_threshold: int | None = None,
        oob_tensor_consumer: OOBTensorConsumer | None = None,
    )
```

Custom encoder with:
- Zero-copy tensor serialization for tensors above `size_threshold`
- Out-of-band (OOB) tensor handling via `OOBTensorConsumer`
- Support for numpy arrays, slices, multimodal items
- Pickle/cloudpickle fallback for arbitrary objects (when `VLLM_ALLOW_INSECURE_SERIALIZATION=1`)

Methods:
- `encode(obj: Any) -> Sequence[bytestr]` - Encode to msgpack with aux buffers
- `encode_into(obj: Any, buf: bytearray) -> Sequence[bytestr]` - Encode into existing buffer
- `enc_hook(obj: Any) -> Any` - Custom encoding hook for non-standard types

### MsgpackDecoder

```python
class MsgpackDecoder:
    def __init__(
        self,
        t: Any | None = None,
        share_mem: bool = True,
        oob_tensor_provider: OOBTensorProvider | None = None,
    )
```

Custom decoder with:
- Zero-copy tensor deserialization with optional pin_memory
- OOB tensor reconstruction via `OOBTensorProvider`
- Automatic type conversion via `dec_hook`

Methods:
- `decode(bufs: bytestr | Sequence[bytestr]) -> Any` - Decode from msgpack buffers
- `dec_hook(t: type, obj: Any) -> Any` - Custom decoding hook for non-standard types
- `ext_hook(code: int, data: memoryview) -> Any` - Extension type handler

### Utility Classes

- `UtilityResult` - Wrapper for special serialization handling
- `PydanticMsgspecMixin` - Makes `msgspec.Struct` compatible with Pydantic validation and serialization
- `OOBTensorConsumer` - ABC for out-of-band tensor consumption
- `OOBTensorProvider` - Callable that reconstructs tensors from OOB data

### Helper Function

```python
def run_method(obj: Any, method: str | bytes | Callable, args: tuple, kwargs: dict) -> Any
```

Runs a method on an object, supporting string method names, serialized bytes (cloudpickle), or direct callables.
