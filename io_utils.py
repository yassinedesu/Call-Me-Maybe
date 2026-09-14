# ABOUTME: Load and validate JSON inputs and write the JSON output, safely.
"""All filesystem access lives here so error handling is in one place."""

import json
import os
from typing import Any

from pydantic import ValidationError

from .models import FunctionDef, OutputItem, TestItem


def _read_json(path: str) -> Any:
    """Read a JSON file, raising a clear ValueError on any problem."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}")


def load_functions(path: str) -> list[FunctionDef]:
    """Load and validate the function definitions."""
    data = _read_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must be a non-empty JSON array of functions")
    try:
        return [FunctionDef.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"Bad function definition in {path}: {exc}")


def load_tests(path: str) -> list[TestItem]:
    """Load and validate the list of prompts."""
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array of prompts")
    try:
        return [TestItem.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"Bad prompt entry in {path}: {exc}")


def save_results(path: str, items: list[OutputItem]) -> None:
    """Write the validated output records to disk as pretty JSON."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    payload = [item.model_dump() for item in items]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
