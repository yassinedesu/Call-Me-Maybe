# ABOUTME: Turn the model's vocab.json into an id->text map plus helper id sets.
"""Maps token ids to the exact characters they add to the output.

Qwen uses byte-level BPE: in vocab.json a leading space is written as the
character 'G-with-dot' (U+0120), and other raw bytes are remapped to printable
characters. We reverse that mapping so we know what real text each token is.
"""

import json
from typing import Optional

import numpy as np

NUMBER_CHARS = set(" 0123456789.-")
STRING_FORBIDDEN = set('"\\\n\r')


def _bytes_to_unicode() -> dict[int, str]:
    """The standard GPT-2/Qwen byte<->unicode table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class TokenMap:
    """Holds id->text and a few precomputed id sets used by the decoder."""

    def __init__(self, id_to_text: dict[int, str]) -> None:
        self.id_to_text = id_to_text
        # id of the token that is exactly a double-quote (used to end strings)
        self.quote_id: Optional[int] = next(
            (i for i, t in id_to_text.items() if t == '"'), None
        )
        # tokens usable inside a JSON number (optional leading space + digits)
        self.number_ids = np.fromiter(
            (
                i
                for i, t in id_to_text.items()
                if t and set(t) <= NUMBER_CHARS and t.strip(" ") != ""
            ),
            dtype=np.int64,
        )
        # tokens usable inside a string (no quote, backslash or newline)
        self.content_ids = np.fromiter(
            (i for i, t in id_to_text.items() if t and not (set(t) & STRING_FORBIDDEN)),
            dtype=np.int64,
        )
        self._charset_cache: dict[frozenset[str], list[tuple[int, str]]] = {}

    @classmethod
    def load(cls, vocab_path: str) -> "TokenMap":
        """Build the map from a vocab.json file path."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)
        decoder = {v: k for k, v in _bytes_to_unicode().items()}
        id_to_text: dict[int, str] = {}
        for token_str, tid in raw.items():
            try:
                text = bytes(decoder[c] for c in token_str).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue  # skip special/undecodable tokens; we never select them
            id_to_text[tid] = text
        return cls(id_to_text)

    def ids_with_charset(self, charset: frozenset[str]) -> list[tuple[int, str]]:
        """All (id, text) whose text uses only `charset` and has no whitespace."""
        if charset not in self._charset_cache:
            self._charset_cache[charset] = [
                (i, t)
                for i, t in self.id_to_text.items()
                if t and set(t) <= charset and " " not in t and "\n" not in t
            ]
        return self._charset_cache[charset]
