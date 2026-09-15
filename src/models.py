from typing import Any
from pydantic import BaseModel, Field


class FunctionSpec(BaseModel):
    """Represents a callable function's schema."""

    name: str
    description: str = ""
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    returns: dict[str, Any] = Field(default_factory=dict)
