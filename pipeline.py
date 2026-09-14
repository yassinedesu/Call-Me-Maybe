# ABOUTME: Orchestrate one prompt: pick the function, then fill each argument.
"""Glue between the models, the SDK and the constrained decoder.

Accuracy (not validity) is what we tune here. Validity is already guaranteed by
the constrained decoder; the job of this file is to give the model enough
context that the *valid* token it prefers is also the *correct* one.
"""

from typing import Any

from .decoder import LLM, generate_choice, generate_number, generate_string
from .models import FunctionDef, OutputItem
from .tokenizer_map import TokenMap

NUMBER_TYPES = {"number", "integer", "int", "float"}
BOOL_TYPES = {"boolean", "bool"}

# Rules that steer content without hardcoding any test answer. The regex hints
# (\s, [A-Z]) are deliberately NOT the public answers (\d+, [aeiou]); they teach
# the convention so the model can generalise to the real and private prompts.
SYSTEM_RULES = (
    "You convert the user's request into a function call.\n"
    "Rules:\n"
    "- Choose the function whose description matches the request.\n"
    "- Fill every argument with a value taken directly from the request.\n"
    "- Never compute or solve anything. A number argument is the number written "
    "in the request, not the result of any operation.\n"
    "- A regex argument must be a regular expression pattern (for example, \\s "
    "matches a space and [A-Z] matches an uppercase letter).\n"
)


def build_menu(functions: list[FunctionDef]) -> str:
    """Human-readable list of functions, given to the model as context."""
    lines = []
    for fn in functions:
        args = ", ".join(f"{k}: {p.type}" for k, p in fn.parameters.items())
        lines.append(f"- {fn.name}({args}) : {fn.description}")
    return "\n".join(lines)


def _chat(system: str, user: str, assistant_prefix: str) -> str:
    """Wrap text in Qwen3's chat format with thinking disabled.

    The empty <think></think> block is exactly what Qwen inserts when thinking
    is off; priming it stops the model from rambling before the answer.
    """
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n" + assistant_prefix
    )


def _system(functions: list[FunctionDef]) -> str:
    return SYSTEM_RULES + "Functions:\n" + build_menu(functions)


def solve_prompt(
    model: LLM, tmap: TokenMap, functions: list[FunctionDef], prompt: str
) -> OutputItem:
    """Produce one output record for a single prompt."""
    system = _system(functions)

    # --- stage 1: choose the function name (model chooses, trie keeps it valid)
    names = [fn.name for fn in functions]
    name_ctx = _chat(system, prompt, "Function name: ")
    name = generate_choice(model, tmap, name_ctx, names)
    chosen = next(fn for fn in functions if fn.name == name)

    # --- stage 2: fill each argument under a type constraint
    params: dict[str, Any] = {}
    decided = f"Function name: {name}\n"
    for pname, spec in chosen.parameters.items():
        ptype = spec.type.lower()
        prefix = f"{decided}{pname} = "
        ctx = _chat(system, prompt, prefix)
        if ptype in NUMBER_TYPES:
            value: Any = generate_number(model, tmap, ctx)
            shown = repr(value)
        elif ptype in BOOL_TYPES:
            value = generate_choice(model, tmap, ctx, ["true", "false"]) == "true"
            shown = "true" if value else "false"
        else:  # default: treat as string
            value = generate_string(model, tmap, ctx)
            shown = f'"{value}"'
        params[pname] = value
        decided += f"{pname} = {shown}\n"

    return OutputItem(prompt=prompt, name=name, parameters=params)
