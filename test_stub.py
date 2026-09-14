# ABOUTME: Verify the constrained decoder with a FAKE model (no torch needed).
"""Scripts a tiny fake LLM so we can prove the decode loop, masking, trie,
number FSM and string FSM behave, without downloading Qwen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decoder import generate_choice, generate_number, generate_string  # noqa: E402
from src.tokenizer_map import TokenMap  # noqa: E402

# A minimal fake vocabulary: id -> real text
VOCAB = {
    0: "fn_add_numbers",
    1: "fn_greet",
    2: '"',
    3: "shrek",
    4: "2",
    5: "3",
    6: " ",
    7: "\n",
    8: "hello",
    9: "16",
    10: ".",
}
VSIZE = 32


class FakeModel:
    """Returns logits that favour whatever the scripted scenario wants next."""

    def __init__(self, favour):
        self.favour = favour  # function(text) -> favoured token id

    def encode(self, text):
        self._text = text

        class T:
            def tolist(_self):
                return [[0]]

        return T()

    def get_logits_from_input_ids(self, input_ids):
        logits = [0.0] * VSIZE
        logits[self.favour(self._text)] = 100.0
        return logits


tmap = TokenMap(VOCAB)


def test_choice_picks_greet():
    model = FakeModel(lambda t: 1)  # always wants fn_greet
    assert generate_choice(model, tmap, "name: ", ["fn_add_numbers", "fn_greet"]) == "fn_greet"


def test_number_stops_after_digit():
    # wants "2" first, then a space (non-number) -> should stop at 2.0
    def favour(t):
        return 4 if not t.rstrip().endswith("2") else 6

    assert generate_number(FakeModel(favour), tmap, "a = ") == 2.0


def test_string_closes_on_quote():
    # wants "shrek", then the closing quote
    def favour(t):
        return 3 if "shrek" not in t else 2

    assert generate_string(FakeModel(favour), tmap, "name = ") == "shrek"


if __name__ == "__main__":
    test_choice_picks_greet()
    test_number_stops_after_digit()
    test_string_closes_on_quote()
    print("all stub tests passed")
