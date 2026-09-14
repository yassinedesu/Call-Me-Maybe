# ABOUTME: Pydantic models validating the input files and the output records.
"""Data models. Every class uses pydantic, as required by the subject."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ParamSpec(BaseModel):
    """One parameter of a function, e.g. {"type": "number"}."""

    type: str


class FunctionDef(BaseModel):
    """One entry of functions_definition.json."""

    name: str
    description: str = ""
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)
    returns: Optional[dict[str, Any]] = None


class TestItem(BaseModel):
    """One entry of function_calling_tests.json."""

    prompt: str


class OutputItem(BaseModel):
    """One record we write to the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
