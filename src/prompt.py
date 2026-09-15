from .models import FunctionSpec


def build_prompt(prompt: str, registry: dict[str, FunctionSpec]) -> str:
    """
    Generates a custom system prompt based on the user query and registry.

    Combines information into a format that allows the model to
    accurately parse and execute intended inputs.

    :param prompt: The input string provided by the user.
    :param registry: Dictionary mapping function names to FunctionSpecs.
    :return: Formatted system prompt string.
    """
    if prompt == "":
        raise ValueError("Prompt cannot be empty.")
    formatted_function: list[str] = []
    for name, spec in registry.items():
        params_list = [
            f"{k}: {v.get('type', 'any')}" for k, v in spec.parameters.items()
        ]
        params_str = ", ".join(params_list)
        formatted_function.append(
            f"- {name}({params_str}): {spec.description}"
        )

    funcs_text = "\n".join(formatted_function)
    return (
        "<|im_start|>system\n"
        "You are a function-calling assistant. Choose the single "
        "best function and extract arguments.\n\n"
        "Rules:\n"
        "- Return ONLY one valid JSON object.\n"
        "- When a parameter expects a regular expression, write valid "
        "Python regex syntax\n"
        "  (e.g. \\d+ for digits, \\bword\\b for whole-word matches, "
        "[aeiou] for character sets).\n"
        "Available functions:\n"
        f"{funcs_text}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
