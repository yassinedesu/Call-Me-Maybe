*This project has been created as part of the 42 curriculum by <your_login>.*

# call me maybe — function calling in LLMs

## Description

This project turns a natural-language prompt into a structured function call.
Given `"What is the sum of 2 and 3?"` it does **not** answer `5`; it returns the
function to call and its typed arguments:

```json
{ "prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0} }
```

Reliability comes from **constrained decoding**: at every generation step we take
the model's raw logits, keep only the tokens that keep the output valid for the
current field, set every other logit to `-inf`, and pick the best remaining one.
The 0.6B model still *chooses* (which function, which values) but can never
produce invalid structure.

## Instructions

```bash
make install          # uv sync (installs numpy, pydantic and the local llm_sdk)
make run              # uv run python -m src  (reads data/input/, writes data/output/)
make lint             # flake8 + mypy
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Algorithm explanation

The output is built in two stages per prompt:

1. **Function selection** — the model is positioned after a short context that
   lists the available functions and the request. We constrain generation to a
   *prefix tree (trie)* of the valid function names: only tokens that keep us on
   the path to a real name survive; the model's logits break the tie between
   names. This satisfies "the function must be chosen by the LLM, not heuristics."
2. **Argument extraction** — once the function is known we know its exact
   parameters and their types. Each value is generated under a type constraint:
   a small automaton for JSON `number` (optional sign, digits, one dot), and a
   quote-terminated automaton for `string`. The final record is assembled as a
   Python object and serialized with `json.dump`, so the file is always valid.

Structural characters (function name is primed with `Function name:`, the string
opening quote is primed) are decided by us, so we only spend a model forward pass
where an actual choice is needed. Token→character mapping uses the model's
`vocab.json` decoded through the GPT-2/Qwen byte-level table (space = `Ġ`).

## Design decisions

- **Two-stage** (name then args) keeps every automaton tiny and explainable.
- **numpy masking** (`-inf` then `argmax`) is the whole constraint in three lines.
- **json.dump** for serialization: values are already type-constrained, so the
  envelope is guaranteed valid without hand-writing JSON.
- **pydantic** validates both inputs and every output record before writing.

## Performance analysis

- *Validity:* 100% by construction — invalid tokens are unpickable.
- *Accuracy:* depends on the model choosing well among valid options; strong on
  arithmetic/greeting/reverse, weaker on complex regex prompts (expected at 0.6B).
- *Speed:* one forward pass only at decision points, so well under 5 minutes.

## Challenges faced

- Tokens are multi-character and byte-level; matching names/values required
  decoding the vocab through the byte table and handling the leading-space `Ġ`.
- Knowing when a number ends: solved by stopping when the model's raw top token
  is no longer a valid number continuation.

## Testing strategy

`tests/test_stub.py` scripts a fake model to verify the trie, number automaton,
string automaton and logit masking without downloading Qwen. Real runs are
validated against `functions_definition.json` and checked for valid JSON output.

## Resources

- Constrained / grammar-guided decoding write-ups and the GPT-2 byte-level BPE
  reference (`bytes_to_unicode`).
- **AI usage:** AI was used to help explain constrained decoding and to draft an
  initial structure of the decoder; all logic was reviewed, rewritten and tested
  by the author, who can explain and modify every part.
