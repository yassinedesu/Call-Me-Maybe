# ABOUTME: Entry point. Parses args, loads inputs, runs the pipeline, saves output.
"""Run with: uv run python -m src [--functions_definition ...] [--input ...] [--output ...]"""

import argparse
import sys

from .io_utils import load_functions, load_tests, save_results
from .models import OutputItem
from .pipeline import solve_prompt
from .tokenizer_map import TokenMap


def parse_args() -> argparse.Namespace:
    """Read the three optional path arguments."""
    parser = argparse.ArgumentParser(description="LLM function-calling with constrained decoding.")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    return parser.parse_args()


def main() -> int:
    """Program entry. Returns a process exit code."""
    args = parse_args()

    try:
        functions = load_functions(args.functions_definition)
        tests = load_tests(args.input)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    # Imported here so input errors above are reported before loading a big model.
    from llm_sdk import Small_LLM_Model

    try:
        model = Small_LLM_Model()
        tmap = TokenMap.load(model.get_path_to_vocab_file())
    except Exception as exc:  # noqa: BLE001 - model/vocab loading must never crash us
        print(f"Model initialisation failed: {exc}", file=sys.stderr)
        return 1

    results: list[OutputItem] = []
    for item in tests:
        try:
            results.append(solve_prompt(model, tmap, functions, item.prompt))
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not stop the rest
            print(f"Skipping prompt {item.prompt!r}: {exc}", file=sys.stderr)

    try:
        save_results(args.output, results)
    except OSError as exc:
        print(f"Could not write output: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
