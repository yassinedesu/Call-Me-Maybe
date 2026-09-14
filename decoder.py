# ABOUTME: Constrained decoding. Mask invalid tokens to -inf, then argmax.
"""The core skill of the project.

At every decision step we take the model's logits, keep only the tokens that
would keep the output valid for the current field, set every other logit to
-inf, and pick the highest remaining one. The model still *chooses*; it just
cannot choose anything invalid.
"""

from typing import Any, Protocol

import numpy as np

from .tokenizer_map import TokenMap


class LLM(Protocol):
    """The subset of the SDK we rely on (public methods only)."""

    def encode(self, text: str) -> Any:
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        ...


def _next_logits(model: LLM, text: str) -> np.ndarray:
    """Tokenise `text` and return the next-token logits as a numpy array."""
    input_ids = model.encode(text).tolist()[0]
    return np.asarray(model.get_logits_from_input_ids(input_ids), dtype=np.float64)


def _argmax_allowed(logits: np.ndarray, allowed: np.ndarray) -> int:
    """Return the id with the highest logit among `allowed` (or -1 if none)."""
    allowed = allowed[allowed < logits.size]
    if allowed.size == 0:
        return -1
    masked = np.full(logits.size, -np.inf)
    masked[allowed] = logits[allowed]
    return int(masked.argmax())


def generate_choice(model: LLM, tmap: TokenMap, context: str, choices: list[str]) -> str:
    """Generate one of `choices` token-by-token using a prefix (trie) constraint."""
    charset = frozenset("".join(choices))
    base = tmap.ids_with_charset(charset)
    current = ""
    for _ in range(100):
        exact = current in choices
        extendable = any(c != current and c.startswith(current) for c in choices)
        if exact and not extendable:
            return current
        logits = _next_logits(model, context + current)
        allowed = np.fromiter(
            (
                tid
                for tid, text in base
                if any(c.startswith(current + text) for c in choices)
            ),
            dtype=np.int64,
        )
        best = _argmax_allowed(logits, allowed)
        if best < 0:
            if exact:
                return current
            raise ValueError("no valid token to continue the choice")
        current += tmap.id_to_text[best]
    raise ValueError("choice did not terminate")


def _number_ok(current: str, piece: str) -> bool:
    """True if appending `piece` still yields a valid JSON-number prefix."""
    if current == "":
        piece = piece.lstrip(" ")
    if piece == "" or " " in piece:
        return False
    s = current + piece
    if s.count(".") > 1 or s.count("-") > 1:
        return False
    if "-" in s and not s.startswith("-"):
        return False
    return all(ch in "0123456789.-" for ch in s)


def generate_number(model: LLM, tmap: TokenMap, context: str) -> float:
    """Generate a JSON number; stop when the model prefers a non-number token."""
    current = ""
    for _ in range(40):
        logits = _next_logits(model, context + current)
        raw_id = int(logits.argmax())
        raw_text = tmap.id_to_text.get(raw_id, "")
        has_digit = any(ch.isdigit() for ch in current)
        if has_digit and not _number_ok(current, raw_text):
            break
        allowed = np.fromiter(
            (tid for tid in tmap.number_ids if _number_ok(current, tmap.id_to_text[tid])),
            dtype=np.int64,
        )
        best = _argmax_allowed(logits, allowed)
        if best < 0:
            break
        piece = tmap.id_to_text[best]
        current += piece.lstrip(" ") if current == "" else piece
    if not any(ch.isdigit() for ch in current):
        raise ValueError("no number produced")
    return float(current)


def generate_string(model: LLM, tmap: TokenMap, context: str, cap: int = 80) -> str:
    """Generate a string value; the opening quote is primed, stop on closing quote."""
    if tmap.quote_id is None:
        raise ValueError("no quote token in vocabulary")
    allowed = np.append(tmap.content_ids, tmap.quote_id)
    content = ""
    primed = context + '"'
    for _ in range(cap):
        logits = _next_logits(model, primed + content)
        best = _argmax_allowed(logits, allowed)
        if best < 0 or best == tmap.quote_id:
            break
        content += tmap.id_to_text[best]
    return content
