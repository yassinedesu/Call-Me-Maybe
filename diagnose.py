# ABOUTME: Dev tool. Runs the REAL model on the public inputs and flags failures.
"""Run:  uv run python -m tests.diagnose

Prints the function + arguments chosen for each public prompt and compares the
function name against the known-correct one, so you can see at a glance which
prompts fail and whether the problem is selection or extraction. Delete or keep
this file; it is not part of the graded pipeline.
"""

from src.io_utils import load_functions, load_tests
from src.pipeline import solve_prompt
from src.tokenizer_map import TokenMap

# Known-correct function name per public prompt (values printed for eyeballing).
EXPECTED_NAME = {
    "What is the sum of 2 and 3?": "fn_add_numbers",
    "What is the sum of 265 and 345?": "fn_add_numbers",
    "Greet shrek": "fn_greet",
    "Greet john": "fn_greet",
    "Reverse the string 'hello'": "fn_reverse_string",
    "Reverse the string 'world'": "fn_reverse_string",
    "What is the square root of 16?": "fn_get_square_root",
    "Calculate the square root of 144": "fn_get_square_root",
}


def main() -> None:
    """Run every public prompt and report."""
    functions = load_functions("data/input/functions_definition.json")
    tests = load_tests("data/input/function_calling_tests.json")

    from llm_sdk import Small_LLM_Model

    model = Small_LLM_Model()
    tmap = TokenMap.load(model.get_path_to_vocab_file())

    for item in tests:
        try:
            out = solve_prompt(model, tmap, functions, item.prompt)
            expected = EXPECTED_NAME.get(item.prompt)
            flag = ""
            if expected and out.name != expected:
                flag = f"  <-- NAME WRONG (expected {expected})"
            print(f"[{out.name}] {item.prompt}")
            print(f"    params = {out.parameters}{flag}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {item.prompt}: {exc}")


if __name__ == "__main__":
    main()
