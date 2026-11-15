# Complete Theory & Code Pipeline — Call Me Maybe

This document maps **how the neural network, tokenizer, and constrained-decoding theory operate step by step** and ties every theoretical mechanism directly to the exact functions and modules in `src/`.

---

## 1. End-to-End Visual Master Diagram (Theory + Code)

```
====================================================================================================
STAGE 0: SCHEMA INGESTION & PROMPT ENGINEERING (CPU / Python)
====================================================================================================
  [data/input/functions_definition.json]
         │
         ▼  src/registry.py :: load_registry()
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Pydantic Boundary Validation (src/models.py :: FunctionSpec)                                │
  │  • Enforces runtime types (name: str, parameters: dict[str, dict[str, Any]])                 │
  │  • Isolates bad definitions via ValidationError without crashing the batch                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼  src/prompt.py :: build_prompt(user_query, registry)
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  ChatML Conditioning Context                                                                 │
  │  "<|im_start|>system\n...Available functions:\n- fn_add_numbers(a: number, b: number)..."    │
  │  "<|im_end|>\n<|im_start|>user\nWhat is the sum of 2 and 3?<|im_end|>\n<|im_start|>assistant\n"
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼  src/pipeline.py :: generator()  [0 Forward Passes — Static Prefix Injection]
  Append literal JSON skeleton opener:  + '{"name":"'

====================================================================================================
STAGE 1: BYTE-LEVEL BPE TOKENIZATION (Text ──► Token IDs)
====================================================================================================
  working_ids = model.encode(full_prompt)[0].tolist()
         │
         ▼  Theoretical Mechanism: 256-Byte Bijection + Subword Merge Rules
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  1. UTF-8 Byte Mapping: Raw bytes (0x00..0xFF) map 1-to-1 to printable Unicode glyphs        │
  │     (e.g., space 0x20 -> 'Ġ' [chr(288)], newline 0x0A -> 'Ċ' [chr(266)]) so there is NEVER   │
  │     an <UNK> (unknown token) error.                                                          │
  │  2. BPE Pair Merging: Adjacent character pairs are greedily merged in priority rank order    │
  │     (merges.txt) until subwords match entries in vocab.json (~151,936 tokens).               │
  │  3. Tensor Unwrap: encode() returns shape [1, T]; [0].tolist() extracts flat list[int].      │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘

====================================================================================================
STAGE 2: INSIDE THE TRANSFORMER DECODER (Token IDs ──► Raw Logits)
  Executed on every call to: logits = model.get_logits_from_input_ids(working_ids)
====================================================================================================
  Input: working_ids = [t_1, t_2, ..., t_T]  (length T)
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  1. Token Embedding Lookup (W_E ∈ ℝ^{|V| × d_model}, where |V| ≈ 151,936, d_model = 1024)     │
  │     • Each integer ID t_i indexes row t_i of W_E ──► continuous vector x_i ∈ ℝ^{1024}        │
  ├──────────────────────────────────────────────────────────────────────────────────────────────┤
  │  2. Stacked Decoder Layers (N = 28 identical blocks in Qwen/Qwen3-0.6B)                      │
  │     For layer l = 1 ... 28:                                                                  │
  │       a) RMSNorm: Normalizes activation scale without mean-centering:                        │
  │          x_norm = (x / sqrt(mean(x^2) + eps)) * gamma                                        │
  │       b) Causal Grouped-Query Attention (GQA) + RoPE:                                        │
  │          • Projects x_norm into Queries (Q), Keys (K), Values (V).                           │
  │          • Rotary Position Embeddings (RoPE) rotate Q_m and K_n by angles m·θ and n·θ so     │
  │            their dot product depends strictly on relative distance (m - n).                  │
  │          • Attention Score: Attention(Q, K, V) = Softmax( (Q·K^T) / sqrt(d_k) + M ) · V      │
  │            where Causal Mask M_{i,j} = 0 if j <= i, and -∞ if j > i (blocks future tokens).  │
  │       c) Residual Connection: x = x + AttentionOut                                           │
  │       d) SwiGLU Feed-Forward MLP:                                                            │
  │          • Expands dimension, applies Swish-gated linear unit: (SiLU(x·W_gate) ⊙ (x·W_up))W_down
  │       e) Residual Connection: x = x + MLPOut                                                 │
  │     *Note on Compute:* Because llm_sdk does not expose a KV-cache, every step recomputes     │
  │     all T positions (O(T^2) attention cost per token).                                       │
  ├──────────────────────────────────────────────────────────────────────────────────────────────┤
  │  3. Final RMSNorm & Language Model Head (W_lm_head ∈ ℝ^{1024 × 151936})                      │
  │     • Takes only the last token's hidden state h_T ∈ ℝ^{1024}                                │
  │     • Linear projection: z = h_T · W_lm_head ──► raw unnormalized logits z ∈ ℝ^{151936}      │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘

====================================================================================================
STAGE 3: CONSTRAINED DECODING LOOP (Logit Masking + Finite-State Automata)
====================================================================================================
  Raw Logits z = [z_0, z_1, ..., z_151935]
         │
         ▼  Grammar / Automaton Validator (src/decoding/*.py)
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Mathematical Guarantee:                                                                     │
  │  In standard generation, P(token_i) = exp(z_i) / sum_j exp(z_j).                             │
  │  By overwriting illegal token logits with -∞ (-math.inf):                                    │
  │     • exp(-∞) = 0  ──► P(illegal token) = 0.000000% (strictly impossible to select)          │
  │     • Because Softmax is strictly monotonic, argmax(z_masked) == argmax(Softmax(z_masked)).  │
  │       We skip Softmax and pick next_id = masked.index(max(masked)) directly (Greedy).        │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──► [3A] Function Name Trie (src/decoding/function_name.py :: select_function_name)
         │     • Pre-encodes all registry keys into token sequences (a Prefix Tree / Trie).
         │     • Allow-list mask: starts with [-math.inf] * |V|, unlocks ONLY next valid trie tokens.
         │     • When generated_tokens == full function name, unlocks closure_id ('"').
         │     • Model's own logits pick among surviving valid function branches!
         │
         ├──► [3B] Decode -> Re-encode BPE Ratchet (src/pipeline.py)
         │     • Python injects structural JSON: '", "parameters": {' and '"<param>": ' (0 model calls).
         │     • Executes: working_ids = model.encode(model.decode(working_ids) + prefix)[0].tolist()
         │     • Prevents non-canonical BPE token splits across concatenated string boundaries.
         │
         └──► [3C] Schema-Typed Parameter Value Selectors (src/decoding/)
               ├── type == "boolean" ──► boolean.py :: select_boolean_value()
               │    • Allow-list mask over exactly 2 IDs: encode("true")[-1] and encode("false")[-1].
               │    • 1 forward pass ──► greedy argmax picks True or False.
               │
               ├── type in ("integer", "number") ──► number.py :: select_number_value()
               │    • Allow-list mask over single-char IDs {'0'..'9', '-', '.'} ∪ stop IDs {'}', ','}.
               │    • Loops up to max_digits=20; breaks when '}' or ',' wins argmax.
               │    • Python casts string to float() or int(float()) with 0/0.0 ValueError fallback.
               │
               └── type == "string" ──► string.py :: select_string_value()
                    • Scans vocab.json to build two sets:
                      1) banned_strings (hard-masked to -∞): '"', '\n', '\r', smart quotes.
                      2) banned_drifts (soft terminators): '{', '}', '.', '\', '\n', "'", schema keys.
                    • Loop breaks on: closing quote '"', drift token (len > 0), or _would_repeat()
                      (detects cyclic n-gram loops of period 1..6).

====================================================================================================
STAGE 4: DETERMINISTIC SERIALIZATION (CPU / Python)
====================================================================================================
  src/pipeline.py returns {"prompt": user_query, "name": function_name, "parameters": parameters}
         │
         ▼  src/__main__.py :: main()
  json.dump(results, f, indent=2) ──► data/output/function_calling_results.json
  *Whole-File Guarantee:* Python writes all braces, colons, commas, and string escapes.
```

---

## 2. Deep-Dive Theory Tied to Code

### 2.1 Why Small LLMs Fail Without Constrained Decoding
An autoregressive language model defines a conditional probability distribution over the next token $x_t$ given context $x_{<t}$:

$$P(x_t \mid x_{<t}) = \text{Softmax}(z_t) = \frac{\exp(z_{t, i})}{\sum_{j=1}^{|V|} \exp(z_{t, j})}$$

In a `0.6B` parameter model (`Qwen/Qwen3-0.6B`), probability mass often leaks onto conversational filler (`"Sure!"`), markdown fences (` ```json `), unclosed quotes, or hallucinated parameter names. Even a $1\%$ per-token error rate across a 30-token JSON payload compounds to a $(1 - 0.99^{30}) \approx 26\%$ failure rate.

**How your codebase eliminates structural failure:**
1. **Structural Offloading (`src/pipeline.py`):** Python never asks the model to predict `{`, `}`, `:`, `,`, `"name"`, `"parameters"`, or parameter key names. Injecting those strings directly saves forward passes and eliminates syntax errors.
2. **Hard Support Truncation (`src/decoding/*.py`):** At true decision points, setting $z_{t, i} = -\infty$ for all $i \notin \mathcal{A}_t$ (where $\mathcal{A}_t$ is the set of valid next tokens from the current automaton state) zeroes out invalid probabilities:

$$\tilde{z}_{t, i} = \begin{cases} z_{t, i} & \text{if } i \in \mathcal{A}_t \\ -\infty & \text{if } i \notin \mathcal{A}_t \end{cases} \implies P(x_t = i \mid x_{<t}) = 0 \quad \forall i \notin \mathcal{A}_t$$

3. **Greedy Argmax (`masked.index(max(masked))`):** Because $\exp(z)$ is strictly increasing, the maximum raw logit corresponds to the maximum post-Softmax probability. Greedy decoding guarantees deterministic, reproducible outputs without sampling variance.

---

### 2.2 Inside `Small_LLM_Model.get_logits_from_input_ids()`
When `src/decoding/` calls `model.get_logits_from_input_ids(working_ids)`, the input list of $T$ integers passes through four neural components:

1. **Embedding Table ($W_E$):** Converts discrete token IDs $[t_1, \dots, t_T]$ into dense continuous vectors $X^{(0)} \in \mathbb{R}^{T \times d_{\text{model}}}$.
2. **Rotary Position Embeddings (RoPE):** Transformers have no built-in notion of token order. RoPE pairs adjacent feature dimensions and rotates Query ($Q$) and Key ($K$) vectors by an angle proportional to token index $m$:
   $$\langle R_m q, R_n k \rangle = q^T R_{n-m} k$$
   This allows attention heads to measure exact relative token distances (e.g., matching `"a"` in the user prompt to parameter `"a"` in the schema).
3. **28 Causal Transformer Blocks:**
   - **Pre-Norm (`RMSNorm`):** Stabilizes activation magnitudes before every sub-layer.
   - **Causal Self-Attention:** Each token attends only to itself and preceding tokens ($j \le i$). Future positions ($j > i$) are masked with $-\infty$ inside the attention matrix before Softmax.
   - **SwiGLU Feed-Forward Network:** Applies non-linear feature transformations independently at each token position, acting as the model's factual/semantic memory store.
4. **Unembedding Projection (`LM Head`):** Only the final position vector $h_T^{(28)}$ is multiplied by $W_{\text{lm\_head}}^T \in \mathbb{R}^{d_{\text{model}} \times |V|}$ to produce the ~151,936 logits for token $T+1$.

---

### 2.3 Byte-Level BPE & Tokenizer Recoding Theory
Understanding how `vocab.json` maps tokens to text is critical both for `src/decoding/string.py` and for recoding the tokenizer without calling `model.encode()` / `model.decode()`:

- **Why Byte-Level BPE?** Character vocabularies create overly long sequences; word vocabularies fail on unseen words (`<UNK>`). Byte-level BPE operates on UTF-8 bytes ($0\dots 255$), guaranteeing every possible string is representable.
- **The 256-Byte Printable Bijection:** Because JSON keys in `vocab.json` must be valid printable UTF-8 strings without raw control characters, BPE maps non-printable/whitespace bytes to shifted Unicode code points (`256 + n`):
  - Byte `0x20` (ASCII space `' '`) $\rightarrow$ `chr(256 + 32)` = `'Ġ'` (`U+0120`)
  - Byte `0x0A` (newline `'\n'`) $\rightarrow$ `chr(256 + 10)` = `'Ċ'` (`U+010A`)
  - Byte `0x09` (tab `'\t'`) $\rightarrow$ `chr(256 + 9)` = `'ĉ'` (`U+0109`)
- **How `decode(ids)` works from `vocab.json` alone:**
  1. Load `vocab.json` (`token_str -> token_id`) via `model.get_path_to_vocab_file()` and invert it once (`id_to_token = {v: k for k, v in vocab.items()}`).
  2. Concatenate `"".join(id_to_token[i] for i in ids)`.
  3. Map each Unicode character back through the inverse 256-byte table into `bytes`, then decode with `.decode("utf-8", errors="replace")`.
- **How `encode(text)` works from `vocab.json` + `merges.txt`:**
  1. Split out special control tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) so they map directly to their reserved IDs.
  2. Convert remaining text to UTF-8 bytes and map each byte to its Unicode stand-in character (`' '` $\rightarrow$ `'Ġ'`).
  3. Iteratively merge the adjacent pair `(a, b)` with the lowest rank in `merges.txt` until no mergeable pairs remain, then look up each resulting subword in `vocab.json`.

---

### 2.4 Why the Decode $\rightarrow$ Re-encode Ratchet Exists in `src/pipeline.py`
In `src/pipeline.py`, structural transitions use:

```python
working_ids = model.encode(model.decode(working_ids) + prefix)[0].tolist()
```

BPE merging is **context-sensitive**: encoding string `A` and string `B` separately (`encode(A) + encode(B)`) does not always yield the same token IDs as `encode(A + B)` because boundary characters may merge into a single subword token (for example, `'"'` followed by `'a'` or `','` followed by `' '`). Re-encoding the concatenated text forces the tokenizer to produce the **canonical token sequence** that `Qwen3-0.6B` observed during pre-training, keeping its hidden states and output logits well-calibrated.

---

## 3. Automaton & Constraint Summary Table

| Pipeline Stage | File & Function | Masking Strategy | Allowed / Banned Set | Termination Trigger |
| :--- | :--- | :--- | :--- | :--- |
| **Function Name** | `src/decoding/function_name.py`<br>`select_function_name()` | **Allow-list** (`[-inf] * V`)<br>Prefix Trie traversal | Next token IDs of registry keys matching `generated_tokens` prefix; `"` when complete | Model selects `closure_id` (`"`) |
| **Boolean Param** | `src/decoding/boolean.py`<br>`select_boolean_value()` | **Allow-list** (`[-inf] * V`)<br>Binary choice | `{encode("true")[-1], encode("false")[-1]}` | Single step (1 forward pass) |
| **Numeric Param** | `src/decoding/number.py`<br>`select_number_value()` | **Allow-list** (`[-inf] * V`)<br>Character-class filter | Single-char IDs in `0-9`, `-`, `.` $\cup$ stop IDs `}`, `,` | Model selects `}` or `,`, or reaches `max_digits=20` |
| **String Param** | `src/decoding/string.py`<br>`select_string_value()` | **Ban-list** (`list(logits)`)<br>Hard ban + soft drift | Hard-bans `"`, `\n`, `\r`, smart quotes; soft-bans `{`, `}`, `.`, `\`, `'`, schema keys | Model selects `"` (`len > 0`), drift token (`len > 0`), `_would_repeat()`, or `max_tokens=30` |

---

## 4. Defense Quick-Reference Checklist

- **How does the LLM choose the function without heuristics?**
  `select_function_name()` builds a prefix trie of all registered function names and masks out every token ID that cannot continue a valid name. Whenever multiple names share a prefix (e.g., `fn_add_numbers` vs. `fn_get_square_root`), all valid next tokens remain unmasked and the model's own forward-pass logits determine the winner.
- **Why set masked logits to `-math.inf` instead of `-100`?**
  In Softmax, $e^{-\infty} = 0$ identically. A finite penalty like `-100` still leaves a non-zero probability and could theoretically win `argmax` if all legal tokens had even lower raw logits (e.g., `-150`).
- **Why does the pipeline make fewer than ~15–20 forward passes per query instead of 40+?**
  Because static JSON syntax (`{"name":"`, `", "parameters": {`, `"param": `, `, `) is injected directly by Python without calling `get_logits_from_input_ids()`. Forward passes are spent strictly at semantic decision points.
- **What guarantees 100% valid JSON output?**
  Two layers: (1) token-level constrained decoding ensures every generated value conforms to its schema type, and (2) `pipeline.py` assembles a native Python `dict` that `__main__.py` serializes via `json.dump()`, making malformed syntax structurally impossible.