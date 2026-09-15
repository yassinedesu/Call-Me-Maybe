# Walkthrough — "call me maybe" (function calling with constrained decoding)

This document explains **what the code does, why each piece exists, and the
concepts a reviewer will probe**. Read it top to bottom once; by the end you
should be able to defend the project and write your own README in your own
words. It is a study/defense aid, not a README you can submit — the section
"Turning this into your README" at the end tells you how to convert it.

> A note on honesty: I could not run the Qwen model while writing this, so
> anything about *empirical* behaviour (accuracy numbers, whether a specific
> prompt succeeds) you should confirm by running it. Everything about *what the
> code does* comes from reading the code and is stated precisely.

---

## 1. The one-paragraph mental model

The model never writes JSON. **Python writes the JSON skeleton** (`{`, `"name":`,
`"parameters": {`, the keys, the commas, the closing braces) and only asks the
model to fill in the *holes*: which function to call, and what value goes in
each parameter slot. Even those holes are filled **one token at a time under
constraints** — at every step we overwrite the model's scores so it can only
pick a token that keeps the output valid. Structure is therefore *guaranteed by
construction*; the model only supplies *content* within the rails we lay down.

That single idea is the whole project. Everything below is the mechanics of it.

---

## 2. Background concepts (know these cold)

### 2.1 Tokens and token IDs
LLMs don't see characters or words; they see **tokens** — subword chunks. A
tokenizer (BPE for Qwen) maps text ↔ a list of integer **token IDs**. `"fn_add"`
might be `["fn", "_add"]` → `[1234, 5678]`. A leading space is part of the token
(the spec's `Ġ` symbol). The vocabulary is a fixed dictionary of ~150k
`token_string → id` entries stored in `vocab.json`.

### 2.2 Logits
For a given input sequence, the model outputs one **logit** (a raw, unnormalised
score) for *every* token in the vocabulary — a list of ~150k floats. Higher =
the model thinks that token is a more likely next token. Softmax would turn
these into probabilities, but we don't need probabilities here: we only need to
compare and pick, so we work on raw logits.

### 2.3 Greedy decoding
"Pick the token with the highest logit, append it, repeat." That's **greedy /
argmax** decoding. This project is fully greedy → deterministic → reproducible
(same input always gives the same output). No sampling, no temperature.

### 2.4 Constrained decoding — the core technique
Between "get logits" and "pick the max", we **edit the logits**:

```python
masked = [-math.inf] * len(logits)   # forbid everything
for tok_id in allowed_tokens:        # re-allow only legal tokens
    masked[tok_id] = logits[tok_id]  # keep their real score
next_id = masked.index(max(masked))  # argmax over what's allowed
```

Because `-inf` can never be the maximum (as long as one allowed token has a
finite score), the chosen token is **always legal**. Crucially, we don't *tell*
the model which legal token to pick — among the allowed ones, its own logits
decide. So the constraint removes *invalid* options; the model still makes the
*choice*. That distinction is what satisfies the spec rule "the function must be
chosen by the LLM, not by heuristics."

There is a mirror-image form (used for strings): start from the *real* logits
and set only the *banned* tokens to `-inf`:

```python
masked = list(logits)
for tok_id in banned:
    masked[tok_id] = -math.inf
```

Use the allow-list form when the set of valid tokens is small and enumerable
(a name, a digit). Use the ban-list form when almost everything is valid and
you only need to forbid a few things (free text).

### 2.5 Why not just prompt the model to "return JSON"?
A 0.6B model asked to emit JSON succeeds maybe ~30% of the time (per the
subject). It forgets a quote, adds prose ("Sure! Here is..."), or invents a
field. Prompting is *hope*; constrained decoding is a *guarantee*. The spec is
explicit that relying on the prompt is not the skill being taught.

---

## 3. The SDK you're given (`llm_sdk.Small_LLM_Model`)

You only get four methods (a fifth, `decode`, is optional but used here):

| Method | Signature | What it gives you |
|---|---|---|
| `encode` | `str -> Tensor` | text → 2-D tensor of token IDs (`[1, n]`) |
| `decode` | `list[int] -> str` | token IDs → text |
| `get_logits_from_input_ids` | `list[int] -> list[float]` | next-token logits for a context |
| `get_path_to_vocab_file` | `() -> str` | path to `vocab.json` (`token → id`) |

You are forbidden from touching private (`_`-prefixed) internals. Everything in
the project is built from these four calls. Two idioms recur:

- `model.encode(text)[0].tolist()` → the tensor's first (only) row as a Python
  `list[int]`. `[0]` drops the batch dimension.
- `model.encode(ch)[0].tolist()[-1]` → the **last** token ID of encoding a short
  string. Taking `[-1]` grabs the meaningful trailing token (e.g. the digit or
  the quote) regardless of any leading-space token the tokenizer prepends.

---

## 4. Architecture and data flow

```
data/input/functions_definition.json   data/input/function_calling_tests.json
                 │                                     │
        registry.py (load+validate)          __main__.py reads prompts
                 │                                     │
                 └──────────────┬──────────────────────┘
                                ▼
                        pipeline.generator(prompt, model, registry)
                                │
              build_prompt() ── prompt.py     (guides the model)
                                │
             select_function_name()  ── decoding/function_name.py
                                │
             per parameter, dispatch on declared type:
               ├─ boolean → decoding/boolean.py
               ├─ number/integer → decoding/number.py
               └─ string  → decoding/string.py
                                │
                                ▼
             {"prompt":..., "name":..., "parameters":{...}}
                                │
                    __main__.py collects → json.dump → data/output/...
```

Two layers guarantee valid output:
1. **Constrained decoding** guarantees each generated *value* is well-formed.
2. **Python assembles the object and `json.dump`s it**, so the *file* is valid
   JSON by construction — the model never produces the braces or commas.

---

## 5. File-by-file

### 5.1 `models.py` — the schema (pydantic)
`FunctionSpec` is a pydantic `BaseModel` with `name`, `description`,
`parameters` (a dict of `param_name → {"type": ...}`), and `returns`. Pydantic
is a spec requirement ("all classes must use pydantic"); it validates the shape
of each function definition for free and raises `ValidationError` on bad data.

### 5.2 `registry.py` — load and validate the function catalogue
Opens the JSON file with a context manager, checks it's a **list**, and builds a
`dict[str, FunctionSpec]`. Malformed *individual* entries are skipped with a
warning (so one bad function doesn't kill the run); a missing file →
`FileNotFoundError`, invalid JSON → `ValueError`. This is the "handle errors
gracefully / never crash" requirement in action.

### 5.3 `prompt.py` — build the system prompt
Formats the available functions into a human-readable list and wraps everything
in Qwen's chat markup (`<|im_start|>system … <|im_end|>` … `assistant`). It also
tells the model to write regex in Python syntax. **Defense point:** this prompt
only *improves the model's choices* (which function, which value). It is **not**
relied on for structure — that's constrained decoding's job. Raising on an empty
prompt is defensive input handling.

### 5.4 `pipeline.py` — the orchestrator (the heart)
`generator()` builds the JSON object hole by hole:

1. `build_prompt(...)` then **append `{"name":"` literally**. The model's context
   now ends mid-JSON, right where a function name must begin.
2. `select_function_name(...)` fills the name under the trie constraint (§6.1).
3. Re-encode `prompt + name + '", "parameters": {'`. Python just wrote the
   structural JSON; the model didn't.
4. For each parameter *in order*:
   - Append the key prefix `"<param>": ` (Python writes it).
   - Dispatch on the **declared type** from the schema:
     - `boolean` → `select_boolean_value` → append `true`/`false`.
     - `integer`/`number` → `select_number_value` → `int(float(x))` or
       `float(x)`; on `ValueError` fall back to `0`/`0.0`.
     - `string` → append a `"`, `select_string_value`, tidy
       (`replace("\\\\","\\").strip()`), append `"<value>"`.
     - anything else → `None`.
   - If not the last parameter, append `, `.
5. Return `{"prompt", "name", "parameters"}`.

**The decode→re-encode "ratchet" — a favourite reviewer question.** Notice the
recurring pattern:

```python
working_ids = model.encode(model.decode(working_ids) + prefix)[0].tolist()
```

Why decode back to text and re-encode, instead of just appending the new token
IDs? Because **BPE tokenisation is context-sensitive**: the canonical tokens for
`"...foo"` + `"bar"` are not always `tokens("...foo") + tokens("bar")` — the
boundary can merge differently. If you feed the model a token sequence it would
never itself produce, its logits degrade. Re-encoding the full string each time
keeps the token stream *canonical*, matching what the model saw in training.
The cost is re-encoding on every step; the payoff is trustworthy logits.

### 5.5 `__main__.py` — the CLI
`argparse` for `--functions_definition`, `--input`, `--output` (with the spec's
default paths). It loads the registry, loads the test cases (guarding
`FileNotFoundError` / `JSONDecodeError`), instantiates the model once, runs every
prompt through `generator` (catching per-prompt exceptions so one failure
doesn't abort the batch), creates the output directory, and `json.dump`s the
results. `KeyboardInterrupt` is caught for a clean Ctrl-C.

### 5.6 `decoding/` — one selector ("validator") per file
`__init__.py` re-exports the four selectors so `pipeline.py` can keep doing
`from .decoding import ...`. The string helpers stay private inside `string.py`.
Splitting by file is purely organisational — the logic is unchanged.

---

## 6. The four selectors in depth

### 6.1 `function_name.py` — constrained name selection (a trie walk)
**Problem:** the name must be *exactly* one registry key. Free generation could
produce `fn_add_number` (missing s) or `add_numbers`. **Solution:** only ever
allow tokens that continue *some* valid name.

```python
valid_sequence = [model.encode(name)[0].tolist() for name in registry]
```

Each valid name becomes its list of token IDs. Then, step by step, tracking the
tokens generated so far:

```python
for seq in valid_sequence:
    if seq[:len(generated_tokens)] == generated_tokens:   # prefix still matches
        if len(seq) > len(generated_tokens):
            allowed_tokens.add(seq[len(generated_tokens)]) # its next token is legal
        else:
            is_complete = True                             # this name fully matched
if is_complete:
    allowed_tokens.add(closure_id)                         # allow the closing quote
```

Mask to `allowed_tokens`, argmax, append; stop when the model picks the closing
`"`. This is effectively **walking a prefix tree (trie) of all valid names**. At
a branch point where several names share the current prefix (e.g.
`fn_add_numbers` vs `fn_add_two`), *multiple* tokens are allowed and the model's
own logits pick the branch — that's the LLM making the decision. The constraint
only guarantees you can never leave the set of real names.

Details worth knowing: `closure_id = encode('"')[0].tolist()[-1]` is the closing
quote token; `max_steps = max(len of any name seq) + 1` bounds the loop; empty
registry raises `ValueError`; the final `decode` strips the closure token.

### 6.2 `number.py` — character-class constraint
Build two token-ID sets by encoding single characters: `allowed` = the 12 chars
`0-9 - .`, `stop` = `}` and `,` (which mean "the number ended"). Loop up to
`max_digits`: mask to `allowed | stop`, argmax; if it's a stop token, break; else
append. Return the decoded numeric string (pipeline converts to `int`/`float`).

Because the allow-set only holds *single-character* tokens, a multi-digit number
like `42` is built digit-token by digit-token (`4`, then `2`). Qwen has
single-digit tokens, so this works; the honest tradeoff is that a hypothetical
multi-char numeric token can't be used. Guaranteed-valid beats maximally-fluent.

### 6.3 `boolean.py` — a one-shot binary decision
A boolean needs no loop — it's one choice. Get the logits once, keep only the
`true` and `false` token logits (everything else `-inf`), argmax, return whether
the winner is the `true` token. The simplest and cleanest selector; use it as
your "explain constrained decoding in 20 seconds" example.

### 6.4 `string.py` — free text with guardrails (the messy one)
Strings are open-ended, so you can't enumerate valid sequences. Instead you use
the **ban-list** form and stop on trouble. Two vocab scans build the ban sets:

- `_load_banned_string_tokens`: bans any token whose decoded text contains
  `"`, `\n`, `\r`, or fancy unicode quotes `” “ ‘ ’`. These would break the JSON
  string or close it early. These are **hard-masked** (never selectable).
- `_load_structural_drift_tokens`: bans tokens containing `'`, `{`, `}`, `.`,
  `\`, `\n`, or the literal words `regex`, `replacement`, `source_string`. These
  are guards against the model "drifting" into structural characters or echoing
  the schema's own parameter names.

Both `discard(close_hammer)` so the legitimate closing quote stays allowed.

The generation loop hard-masks `banned_strings`, then argmaxes, and **breaks**
(ends the string) on any of: (a) chose the closing quote after producing at
least one token; (b) `_would_repeat` fires; (c) chose a *drift* token after
producing at least one token.

Two subtleties reviewers love:
- **`banned_strings` are masked; `banned_drifts` are a soft terminator.** A drift
  token isn't forbidden — if the model *wants* one mid-string, generation simply
  stops there (the `len(generated_tokens) > 0` guard lets a drift token pass only
  as the very first token). So drift characters can't appear inside a captured
  string; they end it.
- **`_would_repeat`** checks periods 1–6: if the last `period` tokens equal the
  `period` tokens immediately before, it's a loop. Tiny models degenerate into
  "hello hello hello"; this is the safety valve.

**The main honest limitation (put this in "Challenges faced").** Because the
drift set bans `\` and `.` (and `'`), a value that legitimately needs those —
above all a **regex** like `\d+` or `[a-z].` for `fn_substitute_string_with_regex`
— will be terminated early or mangled, even though `prompt.py` tells the model to
write regex. The simple cases (`fn_greet` names, `fn_reverse_string` words) are
clean; regex/paths/decimal-in-string values are the weak spot. Verify by running
that specific prompt, and be ready to discuss the tradeoff: the guardrails buy
reliability on the common case at the cost of a few legitimate characters.

---

## 7. Where the two reliability guarantees come from (say this precisely)

1. **Per-token validity** — constrained decoding means every value token is
   legal for its type (a name token, a digit, `true`/`false`, a safe string char).
2. **Whole-file validity** — `pipeline` builds a plain Python dict and
   `__main__` calls `json.dump`. The braces, quotes, commas, and escaping are
   produced by Python's serialiser, not the model, so the file is valid JSON
   with 100% certainty regardless of what the model does.

If asked "what if the model outputs garbage?" the answer is: it *can't* affect
structure — worst case a value is a poor choice (an accuracy issue), never a
malformed-file issue.

---

## 8. Performance and reliability (for your analysis section)

- **Speed per value:** each generated token = one full forward pass
  (`get_logits_from_input_ids`). An N-token value costs N passes. Fine for the
  small test set; greedy so no retries.
- **The obvious inefficiency (good "optimization" bonus material):**
  `_load_banned_string_tokens` and `_load_structural_drift_tokens` **re-read
  `vocab.json` and decode every one of ~150k tokens on every call** to
  `select_string_value`. That repeats for each string parameter of each prompt.
  Compute the ban sets **once** and cache them (module-level, or pass them in) —
  a large, easy win. The re-encode ratchet in §5.4 is the other cost centre.
- **Determinism:** greedy argmax → identical output every run → easy to test and
  to reason about.
- **Accuracy driver:** constraints remove wrong-*structure* options; the model's
  logits still choose *content*, which is where the 90%+ target is won or lost.
  A good prompt (`prompt.py`) helps the model choose well within the rails.

---

## 9. Likely defense questions (with crisp answers)

- **What is a logit?** A raw, pre-softmax score the model assigns to each vocab
  token as the next token. We compare them directly.
- **How does constrained decoding force valid output?** We set forbidden tokens'
  logits to `-inf` before argmax, so only a legal token can win.
- **Why `-inf` and not a big negative number?** `-inf` can *never* be selected;
  a finite penalty could still win if all real options were worse.
- **Is the function chosen by the model or by rules?** By the model. Rules only
  delete invalid names; among valid branches the model's logits pick.
- **Why decode then re-encode in the pipeline?** To keep tokenisation canonical;
  concatenating raw IDs can create sequences the model never trained on and its
  logits become unreliable.
- **What guarantees the output *file* is valid JSON?** Python builds the dict and
  `json.dump` serialises it — the model never writes structural characters.
- **What's the weakest part?** The string selector's heuristic drift bans,
  specifically that banning `\` and `.` breaks legitimate regex values.
- **Empty registry / unknown parameter type?** `select_function_name` raises
  `ValueError`; unknown types default the value to `None`.
- **How would you speed it up?** Cache the vocab-derived ban sets instead of
  recomputing them per string value.
- **Greedy vs sampling — why greedy?** Determinism and reproducibility; for
  structured extraction we want the single best legal token, not diversity.

---

## 10. A tiny end-to-end trace ("What is the sum of 2 and 3?")

1. Prompt built; append `{"name":"`.
2. `select_function_name` walks the trie of registry names; the model's logits
   favour the `fn_add_numbers` branch → name = `fn_add_numbers`.
3. Python writes `", "parameters": {`.
4. Param `a` (type `number`): write `"a": `; `select_number_value` emits `2` (stops
   at the structural boundary) → `float("2")` = `2.0`.
5. Write `, `; param `b`: write `"b": `; number selector emits `3` → `3.0`.
6. Result dict `{"prompt": "...", "name": "fn_add_numbers",
   "parameters": {"a": 2.0, "b": 3.0}}`.
7. `__main__` `json.dump`s it — guaranteed-valid JSON.

---

## 11. Turning this into your README (spec §VI)

The subject requires these README sections; here's where to pull the material
from — **but write it in your own words**, that's the whole point of the defense:

- **Algorithm explanation** → §2.4, §4, §6. Explain constrained decoding and the
  four selectors.
- **Design decisions** → §1 (Python builds structure), §5.4 (ratchet), §6.1
  (trie), §6.4 (ban-list vs allow-list).
- **Performance analysis** → §8.
- **Challenges faced** → §6.4 regex limitation, §5.4 tokenisation boundaries,
  §8 vocab-scan cost.
- **Testing strategy** → describe what *you* did: edge cases from spec §V.6
  (empty strings, big numbers, special characters, multi-parameter functions),
  plus the AST/lint checks and running the sample set.
- **Example usage** → the `uv run python -m src ...` commands and the `Makefile`
  targets.

Also required by the subject: the very first README line, italicised —
`*This project has been created as part of the 42 curriculum by <login>.*` —
and a "Resources" section noting how you used AI (be specific and honest).

---

## 12. Before you defend — a short self-test

Close this file and try to answer without looking:
1. Draw the data flow from the two input files to the output file.
2. Explain, in one sentence, why the output is guaranteed to be valid JSON.
3. Explain how the *function name* is both constrained *and* model-chosen.
4. Name one real limitation of the string selector and why it happens.
5. Explain the decode→re-encode step in the pipeline.

If any answer is shaky, re-read the linked section. When all five feel natural,
you're ready.
