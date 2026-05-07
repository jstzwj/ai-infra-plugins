# SGLang Python Frontend Reference

This document provides a comprehensive reference for the SGLang Python frontend language -- a domain-specific language (DSL) for programming large language models (LLMs). SGLang enables developers to build complex LLM applications using a concise, composable set of primitives that support text generation, structured output, branching, multi-turn conversations, and streaming.

## Table of Contents

1. [Overview](#overview)
2. [Installation and Import](#installation-and-import)
3. [Core Primitives](#core-primitives)
   - [gen()](#gen)
   - [gen_int()](#gen_int)
   - [gen_string()](#gen_string)
   - [select()](#select)
   - [image()](#image)
   - [video()](#video)
4. [Role Management](#role-management)
   - [system(), system_begin(), system_end()](#system-system_begin-system_end)
   - [user(), user_begin(), user_end()](#user-user_begin-user_end)
   - [assistant(), assistant_begin(), assistant_end()](#assistant-assistant_begin-assistant_end)
5. [Control Flow](#control-flow)
   - [function() -- Define Reusable Functions](#function)
   - [Fork and Join (Parallel Branching)](#fork-and-join)
   - [Variable Scoping](#variable-scoping)
   - [Branching on Generated Values](#branching-on-generated-values)
6. [Backend Configuration](#backend-configuration)
   - [RuntimeEndpoint](#runtimeendpoint)
   - [Runtime](#runtime)
   - [Engine](#engine)
   - [set_default_backend()](#set_default_backend)
   - [OpenAI Backend](#openai-backend)
   - [Anthropic Backend](#anthropic-backend)
   - [LiteLLM Backend](#litellm-backend)
   - [VertexAI Backend](#vertexai-backend)
7. [Advanced Features](#advanced-features)
   - [select() with Choices and Scoring Functions](#select-with-choices-and-scoring-functions)
   - [Token Selection Strategies](#token-selection-strategies)
   - [Constrained Generation with Regex and JSON Schema](#constrained-generation)
   - [Streaming Output](#streaming-output)
   - [Batch Execution](#batch-execution)
   - [Multi-turn Conversations](#multi-turn-conversations)
   - [State Management (ProgramState)](#state-management)
   - [separate_reasoning()](#separate_reasoning)
   - [flush_cache()](#flush_cache)
   - [get_server_info()](#get_server_info)
   - [API Speculative Execution](#api-speculative-execution)
8. [Complete API Reference](#complete-api-reference)
9. [Usage Examples](#usage-examples)
10. [Best Practices](#best-practices)

---

## Overview

SGLang (Structured Generation Language) is a frontend language for programming LLMs. It provides a set of Python primitives that can be composed together to build complex LLM applications. The key design principles are:

- **Composability**: Primitives are expressions that can be concatenated with the `+` operator or appended to the state with `+=`.
- **Lazy Evaluation**: SGLang programs are traced into an intermediate representation (IR) and executed by an interpreter that communicates with a backend (local or remote model server).
- **Automatic Optimization**: Common prefixes are automatically cached for batch execution. KV cache operations such as fork and join are handled natively.
- **Backend Agnostic**: Programs can run against SGLang's own runtime server, OpenAI, Anthropic, LiteLLM, or VertexAI backends without modification.

### Programming Model

An SGLang program is a Python function decorated with `@sgl.function`. The first argument is always the state object `s` (a `ProgramState`). Inside the function, the developer appends text, generation calls, role markers, and control flow operations to the state. The runtime executes these operations sequentially, communicating with the backend model.

```
@sgl.function
def my_program(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=128))
```

The decorated function becomes an `SglFunction` object with `run()`, `run_batch()`, `trace()`, and `cache()` methods.

---

## Installation and Import

```python
import sglang as sgl
```

All public APIs are accessible directly from the `sgl` namespace. Key imports include:

```python
from sglang import (
    # Core primitives
    gen, gen_int, gen_string, select, image, video,

    # Role management
    system, system_begin, system_end,
    user, user_begin, user_end,
    assistant, assistant_begin, assistant_end,

    # Function definition
    function,

    # Backend configuration
    Runtime, Engine, RuntimeEndpoint, set_default_backend,

    # Utility functions
    flush_cache, get_server_info, separate_reasoning,

    # Backends
    OpenAI, Anthropic, LiteLLM, VertexAI,

    # Choice methods
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,

    # Configuration
    global_config,
)
```

---

## Core Primitives

### gen()

Generates text by calling the LLM backend. This is the primary primitive for text generation.

```python
sgl.gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    stop_regex: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
    dtype: Optional[Union[type, str]] = None,
    choices: Optional[List[str]] = None,
    choices_method: Optional[ChoicesSamplingMethod] = None,
    regex: Optional[str] = None,
    json_schema: Optional[str] = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` or `None` | `None` | Variable name to store the generated text. If `None`, an auto-generated name is used. Accessed later via `state["name"]`. |
| `max_tokens` | `int` or `None` | `None` | Maximum number of tokens to generate. Overrides the default from `run()`. |
| `min_tokens` | `int` or `None` | `None` | Minimum number of tokens to generate. The model will not emit EOS until this many tokens are produced. |
| `n` | `int` or `None` | `None` | Number of completions to generate. When `n > 1`, the result is a list of strings. |
| `stop` | `str` or `List[str]` or `None` | `None` | Stop sequences. Generation halts when any of these strings is produced. |
| `stop_token_ids` | `List[int]` or `None` | `None` | Stop token IDs. Generation halts when any of these token IDs are produced. |
| `stop_regex` | `str` or `List[str]` or `None` | `None` | Stop when the generated text matches this regular expression. (SGLang runtime only.) |
| `temperature` | `float` or `None` | `None` | Sampling temperature. Higher values increase randomness. Typical range: 0.0--2.0. |
| `top_p` | `float` or `None` | `None` | Nucleus sampling probability. Tokens with cumulative probability above `top_p` are filtered out. |
| `top_k` | `int` or `None` | `None` | Top-k sampling. Only the k most likely tokens are considered. `-1` disables this. |
| `min_p` | `float` or `None` | `None` | Minimum probability relative to the most likely token. Tokens with probability below `min_p * max_prob` are filtered. |
| `frequency_penalty` | `float` or `None` | `None` | Penalty for token frequency. Positive values reduce repetition. Range: -2.0 to 2.0. |
| `presence_penalty` | `float` or `None` | `None` | Penalty for token presence. Positive values encourage new topics. Range: -2.0 to 2.0. |
| `ignore_eos` | `bool` or `None` | `None` | If `True`, the end-of-sequence token is ignored during generation. |
| `return_logprob` | `bool` or `None` | `None` | If `True`, return log probabilities for generated tokens. |
| `logprob_start_len` | `int` or `None` | `None` | Start returning log probabilities from this position in the input. |
| `top_logprobs_num` | `int` or `None` | `None` | Number of top log probabilities to return per token position. |
| `return_text_in_logprobs` | `bool` or `None` | `None` | If `True`, include the text representation of tokens in logprob results. |
| `dtype` | `type` or `str` or `None` | `None` | Constrain output type. Supported: `int`, `float`, `str`, `bool`. Internally converts to regex. |
| `choices` | `List[str]` or `None` | `None` | If provided, automatically converts to a `select()` call. Generates log probabilities for each choice and selects the best one. |
| `choices_method` | `ChoicesSamplingMethod` or `None` | `None` | Method for scoring choices. Default: `token_length_normalized`. Only used when `choices` is set. |
| `regex` | `str` or `None` | `None` | Regular expression for constrained generation. Output will match this regex. (SGLang runtime only.) |
| `json_schema` | `str` or `None` | `None` | JSON schema string for constrained generation. Output will be valid JSON conforming to the schema. (SGLang runtime only.) |

#### Return Value

Returns an `SglGen` expression object (or `SglSelect` if `choices` is provided). When executed, the generated text is stored in the state under the given `name`.

#### Usage Examples

**Basic text generation:**
```python
@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")
```

**Generation with temperature and max tokens:**
```python
s += sgl.gen("story", max_tokens=512, temperature=0.8, top_p=0.95)
```

**Generation with multiple stop sequences:**
```python
s += sgl.gen("output", stop=["\n\n", "END", "---"])
```

**Using choices (automatically becomes select):**
```python
s += "I need to use a " + sgl.gen("tool", choices=["calculator", "search engine"])
```

**Constrained generation with regex:**
```python
s += "IP address: "
s += sgl.gen(
    "ip",
    temperature=0,
    regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
)
```

**Generation with dtype constraint:**
```python
s += "The answer is: " + sgl.gen("number", dtype=int)
```

**Multiple completions (n > 1):**
```python
s += sgl.gen("answers", max_tokens=64, n=3, temperature=1.0)
# state["answers"] is a list of 3 strings
```

**Log probability retrieval:**
```python
s += sgl.gen(
    "text",
    max_tokens=128,
    return_logprob=True,
    top_logprobs_num=5,
    return_text_in_logprobs=True,
)
meta_info = state.get_meta_info("text")
```

**Minimum token enforcement:**
```python
s += sgl.assistant(sgl.gen("answer", min_tokens=64, max_tokens=128))
```

---

### gen_int()

A convenience wrapper around `gen()` that constrains the output to integer values. Internally sets `dtype=int` which uses the regex `[-+]?[0-9]+[ \n]*` and adds space/newline stop tokens.

```python
sgl.gen_int(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    stop_regex: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
)
```

#### Usage Examples

```python
@sgl.function
def math_problem(s, question):
    s += "Question: " + question + "\n"
    s += "Answer: " + sgl.gen_int("result", max_tokens=10)
```

```python
# Generate a rating between 1 and 10
s += "Rate this product (1-10): "
s += sgl.gen_int("rating", max_tokens=2)
```

**Note:** `gen_int()` uses the OpenAI logit bias mechanism for integer tokens when using the OpenAI backend. For SGLang runtime, it uses regex-based constrained generation.

---

### gen_string()

A convenience wrapper around `gen()` that constrains the output to a quoted string value. Internally sets `dtype=str` which uses the regex `"[\w\d\s]*"`.

```python
sgl.gen_string(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    stop_regex: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
)
```

#### Usage Examples

```python
@sgl.function
def name_generator(s, description):
    s += "Generate a name for: " + description + "\n"
    s += "Name: " + sgl.gen_string("name", max_tokens=32)
```

---

### select()

Selects one option from a list of choices by evaluating the log probability of each choice given the prompt context. This is useful for classification, decision-making, and multiple-choice tasks.

```python
sgl.select(
    name: Optional[str] = None,
    choices: Optional[List[str]] = None,
    temperature: float = 0.0,
    choices_method: ChoicesSamplingMethod = token_length_normalized,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` or `None` | `None` | Variable name to store the selected choice. |
| `choices` | `List[str]` or `None` | `None` | **Required.** List of candidate strings to choose from. |
| `temperature` | `float` | `0.0` | Sampling temperature. Must be near 0.0 for the SGLang runtime backend (asserts `temperature <= 1e-5`). |
| `choices_method` | `ChoicesSamplingMethod` | `token_length_normalized` | The method used to score and rank choices. See [Token Selection Strategies](#token-selection-strategies). |

#### Return Value

Returns an `SglSelect` expression. When executed, the selected choice string is stored in the state and appended to the text.

#### Usage Examples

**Simple choice selection:**
```python
@sgl.function
def sentiment_analysis(s, text):
    s += "Text: " + text + "\n"
    s += "Sentiment: " + sgl.select(
        "sentiment",
        choices=["positive", "negative", "neutral"],
    )
```

**Tool selection:**
```python
@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.select(
        "tool",
        choices=["calculator", "search engine", "encyclopedia"],
    )
```

**Note on `gen()` with `choices`:** When you pass `choices` to `gen()`, it internally creates a `select()` call:
```python
# These are equivalent:
sgl.gen("tool", choices=["a", "b", "c"])
sgl.select("tool", choices=["a", "b", "c"], temperature=0.0)
```

---

### image()

Creates an image expression that embeds an image into the prompt. The image is base64-encoded and sent to the backend as part of the request. This is used for multimodal models (e.g., LLaVA, GPT-4V, Gemini).

```python
sgl.image(expr: SglExpr)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `SglExpr` | Typically a string representing the file path to an image, or an `SglArgument` containing a path. Supported formats: JPEG, PNG, etc. |

#### Return Value

Returns an `SglImage` expression object.

#### Usage Examples

**Single image question answering:**
```python
@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))
```

**Image with text context:**
```python
@sgl.function
def image_description(s, image_path):
    s += sgl.user(
        sgl.image(image_path)
        + "\nDescribe what you see in this image in detail."
    )
    s += sgl.assistant(sgl.gen("description", max_tokens=256))
```

---

### video()

Creates a video expression that embeds video frames into the prompt. The video is decoded, sampled into frames, base64-encoded, and sent to the backend.

```python
sgl.video(path: str, num_frames: int)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | File path to the video file. |
| `num_frames` | `int` | Number of frames to sample from the video. |

#### Return Value

Returns an `SglVideo` expression object.

#### Usage Examples

```python
@sgl.function
def video_qa(s, video_path, question):
    s += sgl.user(sgl.video(video_path, num_frames=8) + question)
    s += sgl.assistant(sgl.gen("answer"))
```

---

## Role Management

Role markers define conversation roles (system, user, assistant) and are essential for chat models. They wrap content with the appropriate chat template tokens (e.g., `<|im_start|>user\n...<|im_end|>` for ChatML-based templates).

Each role has three forms:
1. **Inline form** (`system(expr)`, `user(expr)`, `assistant(expr)`): Wraps the given expression with role begin and end markers.
2. **Begin/End form** (`system_begin()`/`system_end()`, etc.): Explicitly marks the beginning and end of a role block.

### system(), system_begin(), system_end()

Sets the system message for the conversation. Typically used once at the beginning of a chat program.

#### Inline Form

```python
sgl.system(expr: Optional[SglExpr] = None)
```

Wraps `expr` in system role markers. If `expr` is `None`, emits empty system role markers.

#### Begin/End Form

```python
sgl.system_begin()  # Marks the start of a system message
sgl.system_end()    # Marks the end of a system message
```

#### Usage Examples

**Inline system message:**
```python
s += sgl.system("You are a helpful assistant.")
```

**Begin/end form:**
```python
s += sgl.system_begin()
s += "You are a helpful assistant."
s += sgl.system_end()
```

**Empty system role (rarely needed):**
```python
s += sgl.system()  # Emits empty begin/end markers
```

---

### user(), user_begin(), user_end()

Sets the user message in the conversation.

#### Inline Form

```python
sgl.user(expr: Optional[SglExpr] = None)
```

#### Begin/End Form

```python
sgl.user_begin()
sgl.user_end()
```

#### Usage Examples

```python
s += sgl.user("What is the capital of France?")
```

```python
s += sgl.user_begin()
s += "Describe this image: "
s += sgl.image("photo.jpg")
s += sgl.user_end()
```

---

### assistant(), assistant_begin(), assistant_end()

Sets the assistant message. Typically used with `gen()` to produce the model's response.

#### Inline Form

```python
sgl.assistant(expr: Optional[SglExpr] = None)
```

#### Begin/End Form

```python
sgl.assistant_begin()
sgl.assistant_end()
```

#### Usage Examples

**Generate assistant response:**
```python
s += sgl.assistant(sgl.gen("answer", max_tokens=256))
```

**Prefill assistant response:**
```python
s += sgl.assistant_begin()
s += "I believe the answer is "
s += sgl.gen("completion", max_tokens=64)
s += sgl.assistant_end()
```

---

## Control Flow

### function()

The `@sgl.function` decorator transforms a Python function into an `SglFunction` object, which can be executed with various backends, batched, streamed, and traced.

```python
@sgl.function
def my_program(s, arg1, arg2, ...):
    # s is the ProgramState
    s += "some text"
    s += sgl.gen("output")
```

Or with explicit `num_api_spec_tokens` for API speculative execution:

```python
@sgl.function(num_api_spec_tokens=128)
def my_program(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))
```

#### SglFunction Methods

The decorated function (`SglFunction`) exposes the following methods:

##### `run()`

Execute the program once.

```python
program.run(
    *args,
    max_new_tokens: int = 128,
    n: int = 1,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    stop_regex: Optional[Union[str, List[str]]] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    ignore_eos: bool = False,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
    stream: bool = False,
    backend=None,
    use_thread: bool = True,
    **kwargs,
) -> ProgramState
```

The sampling parameters provided to `run()` serve as defaults for all `gen()` calls within the program. Individual `gen()` calls can override these defaults.

##### `run_batch()`

Execute the program multiple times with different arguments. Supports multi-threaded execution.

```python
program.run_batch(
    batch_kwargs: Union[List[dict], List[tuple]],
    *,
    max_new_tokens: int = 128,
    n: int = 1,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    stop_regex: Optional[Union[str, List[str]]] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    ignore_eos: bool = False,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
    backend=None,
    num_threads: Union[str, int] = "auto",
    progress_bar: bool = False,
    generator_style: bool = False,
) -> Union[List[ProgramState], Generator[ProgramState]]
```

**Parameters specific to `run_batch()`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_kwargs` | `List[dict]` or `List[tuple]` | (required) | List of argument dictionaries or tuples for each program run. |
| `num_threads` | `int` or `"auto"` | `"auto"` | Number of threads for parallel execution. `"auto"` uses `max(96, cpu_count * 16)`. |
| `progress_bar` | `bool` | `False` | Show a progress bar during batch execution. |
| `generator_style` | `bool` | `False` | If `True`, returns a generator that yields results one by one instead of collecting all results into a list. |

##### `trace()`

Trace the program to build its intermediate representation without executing it.

```python
program.trace(*, backend=None, **kwargs) -> TracerProgramState
```

##### `cache()`

Pre-cache the common prefix of the program.

```python
program.cache(backend=None)
```

##### `bind()`

Create a new function with pre-bound arguments.

```python
bound_program = program.bind(arg1="value1", arg2="value2")
bound_program.run()  # Uses pre-bound values
```

#### Usage Examples

**Basic function definition and execution:**
```python
@sgl.function
def qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")

# Run with default backend
state = qa.run(question="What is the capital of France?")
print(state["answer"])
```

**Batch execution:**
```python
states = qa.run_batch(
    [
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Germany?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)
for s in states:
    print(s["answer"])
```

**Generator-style batch:**
```python
for state in qa.run_batch(
    [{"question": q} for q in questions],
    generator_style=True,
    num_threads=8,
):
    print(state["answer"])
```

**Variable binding:**
```python
@sgl.function
def template(s, name, age):
    s += f"Name: {name}, Age: {age}\n"
    s += "Bio: " + sgl.gen("bio", max_tokens=128)

bound = template.bind(name="Alice")
state = bound.run(age=30)
```

---

### Fork and Join

Fork creates multiple copies of the current state for parallel exploration. Join merges the results back. This is useful for sampling multiple reasoning paths or generating multiple items in parallel.

#### `state.fork()`

```python
state.fork(
    size: int = 1,
    position_ids_offset: Optional[List[int]] = None,
) -> ProgramStateGroup
```

Creates `size` independent copies of the current state. Each copy shares the same prefix (prompt) but can diverge afterward.

#### `ProgramStateGroup.join()`

```python
state_group.join(mode: str = "gather_variable")
```

Merges the forked states back into the source state.

| Mode | Description |
|------|-------------|
| `"gather_variable"` | Collects new variables from each forked state into lists on the source state. Variables that did not exist before the fork become lists of values. |
| `"concate_and_append"` | Concatenates the text from each forked state and appends it. On the SGLang runtime, this uses efficient KV cache concatenation. |

#### `state.copy()`

A context manager that creates a single forked copy and automatically joins on exit.

```python
with state.copy() as s_copy:
    s_copy += sgl.gen("temp_var", max_tokens=64)
# s_copy is joined back automatically
```

#### Usage Examples

**Parallel sampling with fork:**
```python
@sgl.function
def tip_suggestion(s):
    s += "Here are two tips for staying healthy: "
    s += "1. Balanced Diet. 2. Regular Exercise.\n\n"

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += f"Now, expand tip {i+1} into a paragraph:\n"
        f += sgl.gen(f"detailed_tip", max_tokens=256, stop="\n\n")

    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")
```

**Parallel sampling of multiple reasoning paths:**
```python
@sgl.function
def parallel_sample(s, question, n):
    s += "Question: " + question + "\n"
    forks = s.fork(n)
    forks += "Reasoning:" + sgl.gen("reasoning", stop="\n") + "\n"
    forks += "Tool:" + sgl.gen("tool", choices=["calculator", "browser"]) + "\n"
    forks += "Answer:" + sgl.gen("answer", stop="\n") + "\n"
    forks.join()
```

After `join()`, variables from forked states become lists:
```python
state = parallel_sample.run(question="Compute 5 + 2 + 4.", n=5)
for i in range(5):
    print(state["reasoning"][i], state["answer"][i])
```

---

### Variable Scoping

The `var_scope` context manager on the state object captures text generated within a scope into a named variable.

```python
with s.var_scope("my_scope"):
    s += "some text"
    s += sgl.gen("inner_gen")
# s["my_scope"] now contains all text generated within the scope
```

---

### Branching on Generated Values

You can use Python `if` statements to branch based on generated values. The runtime reads the variable value and follows the appropriate path.

```python
@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ". "
    s += (
        "I need to use a "
        + sgl.gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += "The math expression is" + sgl.gen("expression")
    elif s["tool"] == "search engine":
        s += "The key word to search is" + sgl.gen("word")
```

---

## Backend Configuration

### RuntimeEndpoint

Connects to a running SGLang server via HTTP.

```python
sgl.RuntimeEndpoint(
    base_url: str,
    api_key: Optional[str] = None,
    verify: Optional[str] = None,
    chat_template_name: Optional[str] = None,
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_url` | `str` | URL of the SGLang server (e.g., `"http://localhost:30000"`). |
| `api_key` | `str` or `None` | API key for authentication. |
| `verify` | `str` or `None` | SSL certificate verification path. |
| `chat_template_name` | `str` or `None` | Override the chat template. If `None`, auto-detected from the model. |

#### Usage Examples

```python
backend = sgl.RuntimeEndpoint("http://localhost:30000")
sgl.set_default_backend(backend)
```

```python
# With custom chat template
backend = sgl.RuntimeEndpoint(
    "http://localhost:30000",
    chat_template_name="llama-3-instruct",
)
```

---

### Runtime

Launches an SGLang server in a subprocess and connects to it. This is useful for running programs entirely within a Python process without manually starting a server.

```python
sgl.Runtime(
    model_path: str,
    log_level: str = "error",
    launch_timeout: float = 300.0,
    *args,
    **kwargs,
)
```

#### Parameters

The `*args` and `**kwargs` are forwarded to `ServerArgs`. Common parameters include:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | HuggingFace model name or local path to the model. |
| `log_level` | `str` | Log level for the server. Default: `"error"`. |
| `launch_timeout` | `float` | Timeout in seconds for waiting for the server to start. |
| `tp_size` | `int` | Tensor parallelism size. Default: `1`. |
| `mem_fraction_static` | `float` | Fraction of GPU memory for static allocation. |
| `port` | `int` | Port to bind the server to. Auto-selected if not specified. |

#### Methods

| Method | Description |
|--------|-------------|
| `shutdown()` | Shut down the server process. |
| `start_profile()` | Start profiling. |
| `stop_profile()` | Stop profiling. |
| `cache_prefix(prefix)` | Pre-cache a text prefix. |
| `get_tokenizer()` | Get the tokenizer used by the server. |
| `generate(prompt, sampling_params)` | Direct generation call (non-frontend). |
| `encode(prompt)` | Encode text to tokens. |
| `async_generate(prompt, sampling_params)` | Async streaming generation. |

#### Usage Examples

```python
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
sgl.set_default_backend(runtime)

state = my_program.run(question="Hello")
print(state["answer"])

runtime.shutdown()
```

```python
# Multi-GPU
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-70b-chat-hf",
    tp_size=4,
)
```

**Note:** `Runtime` wraps a `RuntimeEndpoint` accessible as `runtime.endpoint`. At interpreter exit, `shutdown()` is called automatically via `atexit`.

---

### Engine

The `Engine` class provides a direct in-process engine without launching an HTTP server. It is intended for offline processing and can be more efficient than the `Runtime` class when HTTP overhead is not needed.

```python
sgl.Engine(*args, **kwargs)
```

Parameters are forwarded to the SRT Engine class. This is imported lazily from `sglang.srt.entrypoints.engine`.

---

### set_default_backend()

Sets the default backend for all subsequent `run()` and `run_batch()` calls.

```python
sgl.set_default_backend(backend: BaseBackend)
```

#### Usage Examples

```python
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))
sgl.set_default_backend(sgl.Anthropic("claude-3-haiku-20240307"))
sgl.set_default_backend(sgl.LiteLLM("anthropic/claude-3-haiku"))
sgl.set_default_backend(sgl.VertexAI("gemini-pro-vision"))
```

---

### OpenAI Backend

Connects to OpenAI's API (or Azure OpenAI).

```python
sgl.OpenAI(
    model_name: str,
    is_chat_model: Optional[bool] = None,
    chat_template: Optional[ChatTemplate] = None,
    is_azure: bool = False,
    *args,
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | OpenAI model name (e.g., `"gpt-3.5-turbo"`, `"gpt-4"`, `"gpt-3.5-turbo-instruct"`). |
| `is_chat_model` | `bool` or `None` | Whether the model uses the chat API. Auto-detected if `None`. Instruct models (`gpt-3.5-turbo-instruct`) use the completions API. |
| `chat_template` | `ChatTemplate` or `None` | Custom chat template. Auto-detected from model name if `None`. |
| `is_azure` | `bool` | Use Azure OpenAI. Default: `False`. |
| `*args, **kwargs` | | Forwarded to `openai.OpenAI()` or `openai.AzureOpenAI()`. Requires `OPENAI_API_KEY` environment variable. |

#### Usage Examples

```python
# Chat model
sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))

# Completion model
sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))

# Azure OpenAI
sgl.set_default_backend(sgl.OpenAI(
    "gpt-4",
    is_azure=True,
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",
    api_version="2024-02-01",
))
```

#### Notes

- For chat models, `sgl.gen()` must appear inside an `assistant()` role block, or API speculative execution must be enabled via `@sgl.function(num_api_spec_tokens=128)`.
- The `select()` method is not supported for chat models. Use a completion model (e.g., `gpt-3.5-turbo-instruct`) for choices.
- `dtype=int` uses logit bias to constrain tokens to digits.
- Regex-constrained generation is not supported (warning is issued, parameter is ignored).

---

### Anthropic Backend

Connects to Anthropic's Claude API.

```python
sgl.Anthropic(
    model_name: str,
    *args,
    **kwargs,
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Anthropic model name (e.g., `"claude-3-haiku-20240307"`, `"claude-3-opus-20240229"`). |
| `*args, **kwargs` | | Forwarded to `anthropic.Anthropic()`. Requires `ANTHROPIC_API_KEY` environment variable. |

#### Usage Examples

```python
sgl.set_default_backend(sgl.Anthropic("claude-3-haiku-20240307"))
```

#### Notes

- Uses the Claude chat template automatically.
- `frequency_penalty` and `presence_penalty` are not supported by the Anthropic API and are dropped.
- Regex-constrained generation is not supported.

---

### LiteLLM Backend

Connects to any LLM provider supported by the LiteLLM library.

```python
sgl.LiteLLM(
    model_name: str,
    chat_template: Optional[ChatTemplate] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = 600,
    max_retries: Optional[int] = None,
    default_headers: Optional[Mapping[str, str]] = None,
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | LiteLLM model identifier (e.g., `"anthropic/claude-3-haiku"`, `"huggingface/meta-llama/Llama-2-7b-chat-hf"`). |
| `chat_template` | `ChatTemplate` or `None` | Custom chat template. Auto-detected if `None`. |
| `api_key` | `str` or `None` | API key for the provider. |
| `organization` | `str` or `None` | Organization for the API. |
| `base_url` | `str` or `None` | Custom base URL for the API. |
| `timeout` | `float` | Request timeout in seconds. Default: `600`. |
| `max_retries` | `int` or `None` | Maximum number of retries. Default: LiteLLM's default. |
| `default_headers` | `Mapping[str, str]` or `None` | Default HTTP headers. |

#### Usage Examples

```python
sgl.set_default_backend(sgl.LiteLLM("anthropic/claude-3-haiku"))
sgl.set_default_backend(sgl.LiteLLM("huggingface/meta-llama/Llama-2-7b-chat-hf"))
```

---

### VertexAI Backend

Connects to Google Cloud Vertex AI's generative models.

```python
sgl.VertexAI(
    model_name: str,
    safety_settings=None,
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | VertexAI model name (e.g., `"gemini-pro-vision"`). |
| `safety_settings` | | Safety configuration for the model. |

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | **Required.** Google Cloud project ID. |
| `GCP_LOCATION` | Google Cloud location/region. |

#### Usage Examples

```python
import os
os.environ["GCP_PROJECT_ID"] = "my-project"
os.environ["GCP_LOCATION"] = "us-central1"
sgl.set_default_backend(sgl.VertexAI("gemini-pro-vision"))
```

---

## Advanced Features

### select() with Choices and Scoring Functions

When you use `select()` or `gen(choices=[...])`, the backend computes the log probability of each choice appended to the current prompt. The choice with the highest score is selected.

For the SGLang runtime backend, this works as follows:
1. The common prefix is cached (generation with 0 new tokens).
2. Each choice is appended to the prompt, and the input token log probabilities are computed.
3. The scoring method ranks the choices based on their log probabilities.

For the OpenAI completion backend, this uses a greedy token-by-token selection with logit bias.

You can retrieve detailed log probability information using `get_meta_info()`:

```python
@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "search engine"])

state = tool_use.run(question="What is 5 + 5?")
meta_info = state.get_meta_info("tool")
print("normalized_prompt_logprobs:", meta_info["normalized_prompt_logprobs"])
print("input_token_logprobs:", meta_info["input_token_logprobs"])
print("output_token_logprobs:", meta_info["output_token_logprobs"])
```

---

### Token Selection Strategies

SGLang provides three built-in strategies for scoring choices:

#### `token_length_normalized` (Default)

Selects the choice with the highest average log probability per token. This normalizes for the fact that longer choices accumulate more negative log probabilities.

```python
from sglang import token_length_normalized

s += sgl.select("choice", choices=["a", "b", "c"], choices_method=token_length_normalized)
```

#### `greedy_token_selection`

Selects based on a greedy token-by-token comparison. For overlapping choices where one is a prefix of another, the shorter choice's average log probability is used to extend it for comparison.

```python
from sglang import greedy_token_selection

s += sgl.select("choice", choices=["yes", "no"], choices_method=greedy_token_selection)
```

#### `unconditional_likelihood_normalized`

Normalizes the log probabilities by the unconditional (context-free) token log probabilities. This requires the backend to compute unconditional log probabilities separately and is more computationally expensive.

```python
from sglang import unconditional_likelihood_normalized

s += sgl.select(
    "choice",
    choices=["positive", "negative", "neutral"],
    choices_method=unconditional_likelihood_normalized,
)
```

---

### Constrained Generation

SGLang supports constraining the output of `gen()` using regular expressions or JSON schemas. This is only supported by the SGLang runtime backend.

#### Regex Constrained Generation

```python
# IP address generation
s += sgl.gen(
    "ip",
    temperature=0,
    regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
)
```

```python
# Complex JSON structure with regex
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""\}"""
)
s += sgl.gen("json_output", max_tokens=256, regex=character_regex)
```

#### JSON Schema Constrained Generation

```python
s += sgl.gen(
    "data",
    max_tokens=128,
    json_schema='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}',
)
```

#### Using Pydantic Models with Regex

SGLang can convert Pydantic models to regex for constrained generation:

```python
from enum import Enum
from pydantic import BaseModel
from sglang.srt.constrained.outlines_backend import build_regex_from_object

class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    bow = "bow"

class Wizard(BaseModel):
    name: str
    age: int
    weapon: Weapon

@sgl.function
def wizard_gen(s):
    s += "Give me a description about a wizard in the JSON format.\n"
    s += sgl.gen(
        "character",
        max_tokens=128,
        temperature=0,
        regex=build_regex_from_object(Wizard),
    )
```

#### dtype Constrained Generation

The `dtype` parameter provides a shorthand for common type constraints:

| dtype | Regex Used | Stop Tokens |
|-------|-----------|-------------|
| `int` | `[-+]?[0-9]+[ \n]*` | `[" ", "\n"]` |
| `float` | `[-+]?[0-9]*\.?[0-9]+[ \n]*` | `[" ", "\n"]` |
| `bool` | `(True\|False)` | -- |
| `str` | `"[\w\d\s]*"` | -- |

---

### Streaming Output

SGLang supports streaming output for both synchronous and asynchronous consumers.

#### Synchronous Streaming

```python
state = program.run(question="What is the capital of France?", stream=True)

# Stream all text
for out in state.text_iter():
    print(out, end="", flush=True)

# Stream a specific variable
for out in state.text_iter(var_name="answer"):
    print(out, end="", flush=True)
```

#### Asynchronous Streaming

```python
state = program.run(question="What is the capital of France?", stream=True)

async for out in state.text_async_iter(var_name="answer"):
    print(out, end="", flush=True)

# With metadata
async for out, meta_info in state.text_async_iter(var_name="answer", return_meta_data=True):
    print(out, end="", flush=True)
```

---

### Batch Execution

Batch execution runs multiple instances of a program in parallel using a thread pool.

```python
@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")

states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for s in states:
    print(s["answer"])
```

#### Batch with Generator Style

For large batches, use `generator_style=True` to yield results as they complete rather than waiting for all results:

```python
for state in text_qa.run_batch(
    [{"question": q} for q in questions],
    generator_style=True,
    num_threads=32,
    progress_bar=True,
):
    process(state)
```

#### Batch Arguments Format

Arguments can be provided as:
- A list of dictionaries: `[{"question": "Q1"}, {"question": "Q2"}]`
- A list of tuples/lists (positional): `[("Q1",), ("Q2",)]`

#### Automatic Prefix Caching

When `global_config.enable_precache_with_tracing` is `True` (the default), SGLang automatically traces programs to extract common prefixes and pre-caches them on the server before batch execution begins.

---

### Multi-turn Conversations

Multi-turn conversations are built by alternating `user()` and `assistant()` role markers. The chat template handles converting these to the model-specific format.

```python
@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))
```

After execution, the full message history is available:

```python
state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

# Access message history
for m in state.messages():
    print(m["role"], ":", m["content"])

# Access individual answers
print(state["answer_1"])
print(state["answer_2"])
```

---

### State Management

The `ProgramState` object is the primary interface for accessing results during and after program execution.

#### ProgramState Methods

| Method | Description |
|--------|-------------|
| `text()` | Returns the full generated text (waits for completion). |
| `messages()` | Returns the message list in OpenAI chat format (waits for completion). |
| `get_var(name)` | Gets the value of a named variable (waits until available). |
| `set_var(name, value)` | Sets a variable value directly. |
| `get_meta_info(name, timeout=None)` | Gets metadata (logprobs, etc.) for a named variable. |
| `text_iter(var_name=None)` | Returns an iterator for streaming text. If `var_name` is provided, streams only that variable's content. |
| `text_async_iter(var_name=None, return_meta_data=False)` | Async version of `text_iter()`. |
| `sync()` | Waits for all pending operations to complete. |
| `error()` | Returns any error that occurred during execution. |
| `fork(size, position_ids_offset=None)` | Forks the state into `size` copies. |
| `copy(position_ids_offset=None)` | Context manager for creating a single forked copy. |
| `var_scope(name)` | Context manager for capturing generated text into a named variable. |

#### Dictionary-like Access

```python
state["answer"]           # Same as state.get_var("answer")
state["answer"] = "text"  # Same as state.set_var("answer", "text")
"name" in state           # Check if variable exists
```

#### Usage Examples

```python
# Get all generated text
full_text = state.text()

# Get message history
messages = state.messages()
# [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]

# Access a variable
answer = state["answer"]

# Get metadata
meta = state.get_meta_info("answer")
if meta:
    print(meta.get("input_token_logprobs"))
    print(meta.get("output_token_logprobs"))
```

---

### separate_reasoning()

Separates reasoning content from the final output. This is designed for models that produce chain-of-thought reasoning (e.g., DeepSeek, QwQ) where the reasoning tokens should be captured separately.

```python
sgl.separate_reasoning(
    expr: Optional[SglExpr] = None,
    model_type: Optional[str] = None,
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `SglExpr` or `None` | The expression (typically a `gen()` or `select()`) whose output should be separated. |
| `model_type` | `str` or `None` | The model type for parsing reasoning content (e.g., `"deepseek"`, `"qwq"`). |

#### How It Works

When applied to a `gen()` call, `separate_reasoning()` parses the output to separate reasoning content (enclosed in think tags) from the normal response. Two variables are created:
- The original `name` variable contains the normal response text.
- A new variable `{name}_reasoning_content` contains the extracted reasoning.

#### Usage Examples

```python
@sgl.function
def reasoning_qa(s, question):
    s += sgl.user(question)
    s += sgl.assistant_begin()
    s += sgl.separate_reasoning(
        sgl.gen("answer", max_tokens=1024),
        model_type="deepseek",
    )
    s += sgl.assistant_end()

state = reasoning_qa.run(question="What is 2 + 2?")
print("Answer:", state["answer"])
print("Reasoning:", state["answer_reasoning_content"])
```

**Note:** `separate_reasoning()` is not supported in streaming mode.

---

### flush_cache()

Flushes the KV cache on the server, freeing GPU memory.

```python
sgl.flush_cache(backend: Optional[BaseBackend] = None) -> bool
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `BaseBackend` or `None` | Backend to flush. If `None`, uses the default backend. |

#### Return Value

Returns `True` if the cache was flushed successfully, `False` if no backend is available.

#### Usage Examples

```python
# Flush default backend
sgl.flush_cache()

# Flush specific backend
sgl.flush_cache(backend=runtime)
```

---

### get_server_info()

Retrieves server information from the SGLang runtime server.

```python
sgl.get_server_info(backend: Optional[BaseBackend] = None) -> dict or None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | `BaseBackend` or `None` | Backend to query. If `None`, uses the default backend. |

#### Return Value

Returns a dictionary with server information, or `None` if no backend is available.

#### Usage Examples

```python
info = sgl.get_server_info()
print(info)
```

---

### API Speculative Execution

For API-based backends (OpenAI chat models), SGLang supports speculative execution to reduce the number of API calls. When enabled via `@sgl.function(num_api_spec_tokens=N)`, the system:

1. Accumulates text and generation calls within an assistant role block.
2. At `assistant_end()`, sends a single API request with enough tokens to satisfy all pending generations.
3. Parses the response to extract individual generation results.

```python
@sgl.function(num_api_spec_tokens=128)
def efficient_chat(s, question):
    s += sgl.user(question)
    s += sgl.assistant(
        sgl.gen("thought", max_tokens=64, stop="\n")
        + sgl.gen("answer", max_tokens=128)
    )
```

This reduces API calls from 2 to 1, trading some accuracy for efficiency.

---

## Complete API Reference

### Expression Objects

All SGLang primitives return expression objects that inherit from `SglExpr`:

| Class | Created By | Description |
|-------|-----------|-------------|
| `SglGen` | `gen()` | Text generation expression |
| `SglSelect` | `select()`, `gen(choices=...)` | Choice selection expression |
| `SglImage` | `image()` | Image embedding expression |
| `SglVideo` | `video()` | Video embedding expression |
| `SglConstantText` | String concatenation | Constant text expression |
| `SglRoleBegin` | `system_begin()`, `user_begin()`, `assistant_begin()` | Role begin marker |
| `SglRoleEnd` | `system_end()`, `user_end()`, `assistant_end()` | Role end marker |
| `SglExprList` | `+` operator, role functions | List of expressions |
| `SglFork` | `state.fork()` | Fork operation |
| `SglVariable` | Internal | Variable reference |
| `SglSeparateReasoning` | `separate_reasoning()` | Reasoning separation |

### Expression Concatenation

Expressions can be concatenated with `+`:

```python
# String + Expression
s += "Hello " + sgl.gen("name")

# Expression + String
s += sgl.gen("name") + " is great."

# Expression + Expression
s += sgl.gen("first") + " " + sgl.gen("last")

# Role wrapping
s += sgl.user("Question: " + sgl.gen("question"))
```

### SglSamplingParams Defaults

When no sampling parameters are specified (either in `run()` or `gen()`), the following defaults apply:

| Parameter | Default |
|-----------|---------|
| `max_new_tokens` | 128 |
| `min_new_tokens` | 0 |
| `n` | 1 |
| `stop` | `()` |
| `stop_token_ids` | `()` |
| `stop_regex` | `()` |
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `top_k` | -1 |
| `min_p` | 0.0 |
| `frequency_penalty` | 0.0 |
| `presence_penalty` | 0.0 |
| `ignore_eos` | False |

### Global Configuration

The `global_config` object controls runtime behavior:

```python
from sglang import global_config

# Verbosity level (0: silent, 2: print output after each run)
global_config.verbosity = 0

# Skip special tokens in output
global_config.skip_special_tokens_in_output = True
global_config.spaces_between_special_tokens_in_out = True

# Enable prefix caching with tracing for batch execution
global_config.enable_precache_with_tracing = True

# Enable parallel encoding for fork/join operations
global_config.enable_parallel_encoding = True
```

---

## Usage Examples

### Example 1: Few-Shot Question Answering (Completion Style)

```python
import sglang as sgl

@sgl.function
def few_shot_qa(s, question):
    s += """The following are questions with answers.
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A: Rome
"""
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)

sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))
state = few_shot_qa.run(question="What is the capital of the United States?")
print(state["answer"])
```

### Example 2: Multi-Turn Chat

```python
import sglang as sgl

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo"))
state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])
```

### Example 3: Streaming Output

```python
import sglang as sgl

@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = text_qa.run(question="What is the capital of France?", temperature=0.1, stream=True)

for out in state.text_iter():
    print(out, end="", flush=True)
print()
```

### Example 4: Batch Execution with Progress Bar

```python
import sglang as sgl

@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n")

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

questions = [
    "What is the capital of the United Kingdom?",
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
]

states = text_qa.run_batch(
    [{"question": q} for q in questions],
    progress_bar=True,
    temperature=0,
)

for q, s in zip(questions, states):
    print(f"{q} -> {s['answer'].strip()}")
```

### Example 5: Parallel Sampling with Fork

```python
import sglang as sgl

@sgl.function
def parallel_sample(s, question, n):
    s += (
        "Question: Compute 1 + 2 + 3\n"
        "Reasoning: I need to use a calculator.\n"
        "Tool: calculator\n"
        "Answer: 6\n"
    )
    s += "Question: " + question + "\n"
    forks = s.fork(n)
    forks += "Reasoning:" + sgl.gen("reasoning", stop="\n") + "\n"
    forks += "Tool:" + sgl.gen("tool", choices=["calculator", "browser"]) + "\n"
    forks += "Answer:" + sgl.gen("answer", stop="\n") + "\n"
    forks.join()

sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))
state = parallel_sample.run(question="Compute 5 + 2 + 4.", n=5, temperature=1.0)

for i in range(5):
    print(f"[{i}] reasoning={state['reasoning'][i]}, tool={state['tool'][i]}, answer={state['answer'][i]}")
```

### Example 6: Tool Use with Branching

```python
import sglang as sgl

@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ". "
    s += (
        "I need to use a "
        + sgl.gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += "The math expression is" + sgl.gen("expression")
    elif s["tool"] == "search engine":
        s += "The key word to search is" + sgl.gen("word")

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = tool_use.run(question="What is the capital of the United States?")
print(state.text())
print("Tool selected:", state["tool"])
```

### Example 7: Constrained JSON Generation with Regex

```python
import sglang as sgl

character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)

@sgl.function
def character_gen(s, name):
    s += (
        name
        + " is a character in Harry Potter. Please fill in the following information.\n"
    )
    s += "The JSON output is:\n"
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = character_gen.run(name="Hermione Granger")
print(state["json_output"])
```

### Example 8: Vision (Multimodal) with LLaVA

```python
import sglang as sgl

@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))

runtime = sgl.Runtime(model_path="lmms-lab/llama3-llava-next-8b")
sgl.set_default_backend(runtime)

state = image_qa.run(
    image_path="images/cat.jpeg",
    question="What is this?",
    max_new_tokens=128,
)
print(state["answer"])

runtime.shutdown()
```

### Example 9: Local Runtime with Streaming and Batch

```python
import sglang as sgl

@sgl.function
def few_shot_qa(s, question):
    s += """Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
"""
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)

runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
sgl.set_default_backend(runtime)

# Single run
state = few_shot_qa.run(question="What is the capital of the United States?")
print(state["answer"])

# Streaming
state = few_shot_qa.run(
    question="What is the capital of the United States?",
    stream=True,
)
for out in state.text_iter("answer"):
    print(out, end="", flush=True)
print()

# Batch
states = few_shot_qa.run_batch(
    [
        {"question": "What is the capital of the United States?"},
        {"question": "What is the capital of China?"},
    ]
)
for s in states:
    print(s["answer"])

runtime.shutdown()
```

### Example 10: Choices with Log Probability Inspection

```python
import sglang as sgl

@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "search engine"])

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

# Single run
state = tool_use.run(question="What is 5 + 5?")
print("Choice:", state["tool"])

meta_info = state.get_meta_info("tool")
print("Logprobs of choice 1:", meta_info["input_token_logprobs"][0])
print("Logprobs of choice 2:", meta_info["input_token_logprobs"][1])

# Batch run
questions = ["What is 5 + 6?", "Who is Michael Jordan?"]
states = tool_use.run_batch([{"question": q} for q in questions])
for question, state in zip(questions, states):
    print(f"Question: {question}")
    print(f"Choice: {state['tool']}")
    meta_info = state.get_meta_info("tool")
    for i, lp in enumerate(meta_info["input_token_logprobs"]):
        print(f"  Logprobs of choice {i+1}:", lp)
```

### Example 11: Chain-of-Thought Decoding

```python
import sglang as sgl

@sgl.function
def cot_decoding(s, question, get_top_k):
    s += sgl.user("Question: " + question + "\nAnswer:")
    s += sgl.assistant_begin()

    step_0 = s.fork(1)[0]
    forks = s.fork(get_top_k)

    # Get top-k tokens at step 0
    step_0 += sgl.gen(
        "get_top_k",
        max_tokens=0,
        return_logprob=True,
        top_logprobs_num=get_top_k,
        return_text_in_logprobs=True,
    )
    logprobs = step_0.get_meta_info("get_top_k")["output_top_logprobs"][0]

    # Explore each top-k path
    for idx, (f, token) in enumerate(zip(forks, logprobs)):
        logprob, token_id, text = token
        f += text
        f += sgl.gen(
            "answer",
            temperature=0,
            max_tokens=1024,
            return_logprob=True,
        )

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = cot_decoding.run(
    question="Claire makes a 3 egg omelet every morning. How many eggs in 4 weeks?",
    get_top_k=5,
)
```

### Example 12: Minimum Token Generation

```python
import sglang as sgl

@sgl.function
def long_answer(s):
    s += sgl.user("Explain quantum computing in simple terms.")
    s += sgl.assistant(sgl.gen("answer", min_tokens=64, max_tokens=128))

runtime = sgl.Runtime(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
sgl.set_default_backend(runtime)

state = long_answer.run()
print(state["answer"])
# The answer will always be at least 64 tokens long

runtime.shutdown()
```

### Example 13: Multiple Completions (n > 1)

```python
import sglang as sgl

@sgl.function
def multi_answer(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answers", max_tokens=1024, n=3))

sgl.set_default_backend(sgl.OpenAI("o1"))
state = multi_answer.run(question="Explain gravity in simple terms.")

# state["answers"] is a list of 3 strings
assert isinstance(state["answers"], list)
assert len(state["answers"]) == 3
for i, ans in enumerate(state["answers"]):
    print(f"Answer {i+1}:", ans)
```

### Example 14: Using Anthropic Backend

```python
import sglang as sgl

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

sgl.set_default_backend(sgl.Anthropic("claude-3-haiku-20240307"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])
```

### Example 15: Chinese Regex Constrained Generation

```python
import sglang as sgl

character_regex = (
    r"""\{\n"""
    + r"""    "姓名": "[^"]{1,32}",\n"""
    + r"""    "学院": "(格兰芬多|赫奇帕奇|拉文克劳|斯莱特林)",\n"""
    + r"""    "血型": "(纯血|混血|麻瓜)",\n"""
    + r"""\}"""
)

@sgl.function
def character_gen(s, name):
    s += name + " is a character in Harry Potter. Fill in the following information."
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = character_gen.run(name="Hermione Granger", temperature=0)
print(state.text())
```

---

## Best Practices

### 1. Always Set a Backend Before Running Programs

SGLang requires a backend to be set either via `set_default_backend()` or passed explicitly to `run()`.

```python
# Good
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = my_program.run()

# Also good
backend = sgl.RuntimeEndpoint("http://localhost:30000")
state = my_program.run(backend=backend)
```

### 2. Use `temperature=0` for Deterministic Outputs

When you need reproducible results, set temperature to 0:

```python
state = program.run(question="...", temperature=0)
# Or per-gen:
s += sgl.gen("answer", temperature=0, max_tokens=128)
```

### 3. Leverage Fork for Parallel Exploration

Fork is more efficient than running multiple independent programs because the prompt prefix is computed only once and shared across all branches via KV cache.

```python
# Good: shared prefix
forks = s.fork(5)
forks += sgl.gen("answer", stop="\n")
forks.join()
```

### 4. Use `run_batch()` for Multiple Inputs

`run_batch()` is optimized for processing multiple inputs, with automatic prefix caching and thread pool parallelism:

```python
# Good: batch execution
states = program.run_batch([{"arg": v} for v in values], progress_bar=True)

# Avoid: sequential runs in a loop
states = []
for v in values:
    states.append(program.run(arg=v))
```

### 5. Use Regex/JSON Schema for Structured Output

When using the SGLang runtime, constrained generation ensures output validity:

```python
# Guaranteed valid JSON matching the schema
s += sgl.gen("data", max_tokens=256, json_schema=schema_string)

# Guaranteed valid regex match
s += sgl.gen("ip", regex=r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
```

### 6. Choose the Right Backend

| Use Case | Recommended Backend |
|----------|-------------------|
| Local GPU inference | `Runtime` or `RuntimeEndpoint` |
| OpenAI models | `OpenAI` |
| Claude models | `Anthropic` |
| Any provider via LiteLLM | `LiteLLM` |
| Gemini models | `VertexAI` |
| Offline processing (no HTTP) | `Engine` |

### 7. Handle `n > 1` Results as Lists

When using `n > 1` in `gen()`, the result is a list. When used with `fork()` and `join()`, forked variables also become lists:

```python
s += sgl.gen("answers", n=3)
# state["answers"] is ["ans1", "ans2", "ans3"]

forks = s.fork(2)
forks += sgl.gen("answer")
forks.join()
# state["answer"] is ["ans1", "ans2"]
```

### 8. Use Chat Templates for Chat Models

When using chat models (GPT-3.5-turbo, Claude, etc.), always wrap content in role markers:

```python
# Good: proper role markers
s += sgl.user(question)
s += sgl.assistant(sgl.gen("answer"))

# Bad: raw text with chat models (may not work correctly)
s += question
s += sgl.gen("answer")
```

### 9. Shut Down Runtimes When Done

Always call `shutdown()` on `Runtime` objects when finished:

```python
runtime = sgl.Runtime(model_path="...")
try:
    # Use the runtime
    state = program.run(...)
finally:
    runtime.shutdown()
```

Note: `Runtime` registers `shutdown()` with `atexit`, so it will be called automatically when the Python process exits. However, explicit shutdown is recommended for long-running applications.

### 10. Use API Speculative Execution for OpenAI Chat Models

For OpenAI chat models, enabling speculative execution can significantly reduce API calls:

```python
@sgl.function(num_api_spec_tokens=128)
def efficient_program(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))
```

### 11. Avoid Nested Roles

Nested roles are not allowed. Always close one role before starting another:

```python
# Good
s += sgl.user("question")
s += sgl.assistant(sgl.gen("answer"))

# Bad: nested roles (raises assertion error)
s += sgl.user_begin()
s += sgl.system("system prompt")  # Error: role already active
s += sgl.user_end()
```

### 12. Do Not Use f-strings with Arguments

SGLang arguments cannot be used inside f-strings because they are traced, not evaluated immediately:

```python
# Bad: raises TypeError
s += f"The answer is {s['answer']}"

# Good: use string concatenation
s += "The answer is " + s["answer"]
```

### 13. Use `select()` for Classification Tasks

For classification and decision-making tasks, `select()` is more efficient than generating free-form text and parsing it:

```python
# Good: efficient classification
s += "Sentiment: " + sgl.select("sentiment", choices=["positive", "negative", "neutral"])

# Inefficient: generate and parse
s += "Sentiment: " + sgl.gen("sentiment", stop="\n")
# then parse state["sentiment"]
```

### 14. Control Memory with `flush_cache()`

For long-running applications, periodically flush the KV cache to free GPU memory:

```python
# After a batch of requests
sgl.flush_cache()
```

### 15. Choose the Right Choices Method

- Use `token_length_normalized` (default) for most cases. It handles length differences fairly.
- Use `greedy_token_selection` when choices may overlap (e.g., "yes" vs "yes, please").
- Use `unconditional_likelihood_normalized` when you need probability estimates that are normalized for inherent token frequency, at the cost of extra computation.

### 16. Streaming with Specific Variables

When streaming, you can choose to stream only a specific variable or the entire text:

```python
# Stream everything
for chunk in state.text_iter():
    print(chunk, end="")

# Stream only a specific variable
for chunk in state.text_iter(var_name="answer"):
    print(chunk, end="")
```

### 17. Use `min_tokens` for Long Responses

When you need the model to produce a response of minimum length, use `min_tokens`:

```python
s += sgl.gen("essay", min_tokens=256, max_tokens=512)
```

The model will not produce an end-of-sequence token until at least 256 tokens are generated.

### 18. Backend-Specific Limitations

Be aware of backend-specific limitations:

| Feature | SGLang Runtime | OpenAI | Anthropic | LiteLLM | VertexAI |
|---------|---------------|--------|-----------|---------|----------|
| Regex constrained gen | Yes | No (warning) | No (warning) | No (warning) | No |
| JSON schema gen | Yes | No | No | No | No |
| `select()` / `choices` | Yes | Yes (completion only) | No | No | No |
| Streaming | Yes | Yes | Yes | Yes | Yes |
| Fork/Join with KV cache | Yes | No | No | No | No |
| Log probabilities | Yes | Limited | No | No | No |
| Vision | Yes | Yes | No | Limited | Yes |
| `stop_regex` | Yes | No | No | No | No |
| `top_k` | Yes | No | Yes | No | Yes |
| `min_p` | Yes | No | No | No | No |
| `min_tokens` | Yes | No | No | No | No |
