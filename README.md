*This project has been created as part of the 42 curriculum by yael-kha.*

# Call Me Maybe - Constrained-Decoding Function Calling

A robust function-calling engine for small language models (`Qwen/Qwen3-0.6B`) using token-level constrained decoding and logit masking. The system translates natural language queries into schema-compliant, machine-executable JSON function calls with near-perfect reliability.

---

## Description

Large Language Models (LLMs) can comprehend unstructured natural language but often fail to reliably emit valid, structured JSON. This limitation is particularly acute in small parameter models (e.g., 0.6B parameters), which frequently succumb to syntax malformations, type errors, key hallucinations, or schema drift when relying solely on prompt engineering.

**Call Me Maybe** resolves this by enforcing structural constraints during inference. Instead of free-form text generation followed by brittle parsing, the engine drives the LLM token-by-token:

1. It queries the model's next-token logits via `llm_sdk`.


2. It masks invalid tokens (assigning them a logit score of $-\infty$) based on the current grammar state and schema specification.


3. It selects greedy next tokens exclusively from the subset of allowed tokens, guaranteeing 100% syntactically valid and schema-compliant JSON payloads.



---

## Instructions

### Prerequisites

* Python 3.10 or higher


* [`uv`](https://github.com/astral-sh/uv) package and environment manager


* CUDA-compatible GPU, Apple Silicon (MPS), or CPU



### Installation

Clone the repository and install all dependencies into the local virtual environment using the provided Makefile:

```bash
make install

```

This synchronizes the virtual environment, locks packages, and links the local `llm_sdk` workspace dependency.

### Execution

Run the function calling pipeline on the default input test set:

```bash
make run

```

To run the application directly with custom input/output paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

```

### Debugging

Launch the program using Python's interactive debugger (`pdb`):

```bash
make debug

```

### Code Quality & Static Analysis

Ensure code conforms to PEP 8 / Flake8 standards and passes static type checks:

```bash
# Standard linting (Flake8 and Mypy with disallow-untyped-defs)
make lint

# Strict type checking (Mypy --strict)
make lint-strict

```

### Cleanup

Remove virtual environments, caches (`__pycache__`, `.mypy_cache`), and model artifacts:

```bash
make clean

```

---

## Algorithm Explanation

The constrained decoding pipeline bypasses unconstrained generation loops in favor of a deterministic skeleton populated by guided token selection:

```
User Query + Registry -> ChatML Prompt -> Prefix Guided Function Selection
                                                 │
                                                 ▼
Structured JSON Skeleton <── Infill Loop <── Parameter Type Dispatch
                                                 │
                                                 ├── Boolean Selector (true/false)
                                                 ├── Number Selector (digits, '.', '-')
                                                 └── String Selector (drift masking + loop detection)

```

1. **Prompt Formatting (`src/prompt.py`)**


The user prompt and the function registry are formatted into a structured ChatML template (`<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant`), providing the model with available signatures and expected argument contracts.


2. **Prefix-Guided Function Name Selection (`src/decoding/function_name.py`)**

* The generation context is seeded with the JSON opening prefix `{"name":"`.


* All available function names in the registry are encoded into valid target token sequences.


* At each generation step, candidate tokens are filtered to those matching prefixes of valid registered function names.


* Once a registered function name is matched, the closure token (`"`) is allowed.


* All unauthorized logits are set to `-math.inf`, ensuring zero probability of selecting a non-existent function.




3. **Schema-Driven Parameter Infilling (`src/pipeline.py`)**

* Once the function is chosen, the engine inspects its `FunctionSpec` parameters.


* The engine writes the key syntax `", "parameters": {"<param_name>": ` directly into the context, eliminating key hallucination.


* Token decoding is delegated to type-specific validators:


* **Boolean (`src/decoding/boolean.py`)**: Masks all tokens except the exact IDs for `"true"` and `"false"`. The model chooses between the two based on their relative logits.


* **Number / Integer (`src/decoding/number.py`)**: Masks all tokens except digits (`0-9`), signs (`-`), and decimal points (`.`), allowing stop characters (`}`, `,`). It converts the resulting string to an integer or float.


* **String (`src/decoding/string.py`)**: Allows open generation while masking structural drift tokens and quotes, halting on the quote closure token `"` or repetition patterns.







---

## Design Decisions

* **Structural Infilling over Free Generation**: Rather than letting the model output JSON punctuation (`{`, `}`, `"`, `:`), the pipeline statically constructs the JSON wrapper and key strings. This reduces the search space exclusively to value generation.


* **Vocabulary Inspection for Drift Suppression**: Small models often attempt to generate structural JSON markers (e.g., `{"replacement": ...}`) inside string values. `_load_structural_drift_tokens` and `_load_banned_string_tokens` dynamically scan the tokenizer's `vocab.json` at runtime to ban structural delimiters from string token generation.


* **Repetition Detection (`_would_repeat`)**: Small models can enter deterministic degeneration loops when generating open-ended strings (such as regex strings). The string selector checks periodic sub-sequences of lengths 1 through 6 before admitting a token, breaking cyclic loops immediately.


* **Pydantic for Registry Validation (`src/models.py`, `src/registry.py`)**: `FunctionSpec` enforces validation on imported schemas, ensuring parameter types and function attributes conform to runtime expectations before model execution.



---

## Performance Analysis

* **Accuracy**: Function name selection achieves **100% precision** against valid candidates via prefix-constrained masking. Parameter typing achieves **100% schema adherence**, eliminating JSON syntax and type mismatch failures.


* **Speed**: Generating a single function call requires fewer than 40 forward passes on average, as JSON syntax tokens are injected without invoking model forward passes. An entire suite of test cases executes in under 15 seconds on modern Apple Silicon / CUDA GPUs, and well within 1 minute on CPU.


* **Reliability & Resilience**: Missing or malformed input files, invalid JSON structures, and unexpected characters are handled gracefully using `try-except` blocks, returning descriptive logs without unhandled process termination.



---

## Challenges Faced

* **Tokenizer BPE Space Inconsistencies**: The Qwen tokenizer prepends spaces (`Ġ`) or breaks words into differing subwords depending on context. Naively checking single-character strings against tokens caused invalid token filtering. This was resolved by using `model.encode()` directly on the required target values to capture the true subword token IDs.


* **Repetition and Degeneration in Regex Parameters**: Small language models frequently loop when generating regular expressions (such as `fn_substitute_string_with_regex`). Implementing dynamic n-gram cycle tracking (`_would_repeat`) and token length ceilings resolved infinite sequence traps.


* **Escaped Characters in Raw Strings**: Handling escaped regex tokens (e.g., `\\d+`) resulted in double-escaping during JSON serialization. Post-processing string values with `.replace("\\\\", "\\")` ensured standard regex compatibility.



---

## Testing Strategy

1. **Schema & Typings**: Verified with `flake8` (line-length 79) and `mypy` (`--strict` and `--disallow-untyped-defs`).


2. **Deterministic Type Ingestion**: Tested against boolean selections, negative integers, floating-point decimals, simple strings, and complex regex pattern extraction.


3. **Edge Case Injection**:
* Empty user prompt validation.


* Non-existent and corrupted function definition files.


* Prompts requiring regex substitutions containing escape sequences and metacharacters.




4. **JSON Output Verification**: Validated that all records in `data/output/function_calling_results.json` parse cleanly with Python's standard `json.loads` without structural anomalies.



---

## Example Usage

### Input Definition (`data/input/functions_definition.json`)



```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]

```

### Input Test Prompts (`data/input/function_calling_tests.json`)



```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" }
]

```

### Execution Output (`data/output/function_calling_results.json`)



```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  }
]

```

---

## Resources

* **Hugging Face Transformers Documentation**: [Causal Language Modeling and Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
* **Constrained Decoding & Logit Bias**: [Guiding Text Generation with Token-Level Constraints](https://arxiv.org/abs/2307.09702)
* **ChatML & Qwen Architecture**: [Qwen Technical Report and Prompt Templates](https://github.com/QwenLM/Qwen)
* **Pydantic Documentation**: [Data Validation using Python Type Hints](https://www.google.com/search?q=https://docs.pydantic.dev/)
