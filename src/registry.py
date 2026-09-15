import json

from pydantic import ValidationError

from .models import FunctionSpec


def load_registry(path: str) -> dict[str, FunctionSpec]:
    """Load and parse the function registry from a JSON file.

    Args:
        path: Path to the functions_definition.json file.

    Returns:
        Mapping of function name to its FunctionSpec.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file contains invalid JSON.
    """
    try:
        with open(path, "r") as config_file:
            function_definitions = json.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Functions definition file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    if not isinstance(function_definitions, list):
        raise ValueError(
            f"Invalid JSON in {path}: Expected a list of functions"
        )

    registry: dict[str, FunctionSpec] = {}
    for func in function_definitions:
        try:
            spec = FunctionSpec(**func)
            registry[spec.name] = spec
        except ValidationError as e:
            print(f"Warning: Skipping invalid function definition: {e}")
            continue
    return registry
