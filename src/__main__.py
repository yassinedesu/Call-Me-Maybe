import argparse
import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .pipeline import generator
from .registry import load_registry


def main() -> None:
    """Entry point: read prompts, run the pipeline, write results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        type=str
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        type=str
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json", type=str
    )
    args = parser.parse_args()

    try:
        registry = load_registry(args.functions_definition)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading function definitions: {e}")
        return
    try:
        with open(args.input, "r") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in input file: {e}")
        return

    model = Small_LLM_Model()
    results = []

    for case in test_cases:
        if not isinstance(case, dict):
            continue
        prompt = case.get("prompt", "")
        try:
            result = generator(prompt, model, registry)
            print(result)
        except Exception as e:
            print(f"Warning: failed to process prompt '{prompt}': {e}")
            continue
        results.append(result)

    # this one creates the directory for the file
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
