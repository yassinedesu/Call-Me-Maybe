from typing import Any

from llm_sdk import Small_LLM_Model

from .decoding import (
    select_boolean_value,
    select_function_name,
    select_number_value,
    select_string_value,
)
from .models import FunctionSpec
from .prompt import build_prompt


def generator(
        user_query: str,
        model: Small_LLM_Model,
        registry: dict[str, FunctionSpec],
) -> dict[str, Any]:
    """
    Generates a structured payload for invoking a function based on the
    user-provided query and a registry of function specifications. This
    function constructs a model prompt, infers the function to call, and
    resolves its parameters by leveraging the provided language model.

    :param user_query: The input query from the user as a string.
    :type user_query: str
    :param model: The instance of a small language model used for
        encoding, decoding, and value selection.
    :type model: Small_LLM_Model
    :param registry: A dictionary mapping function names to their
        respective specifications, which include parameters and metadata.
    :type registry: dict[str, FunctionSpec]
    :return: A dictionary containing the original query, the selected
        function's name, and its resolved parameters.
    :rtype: dict[str, Any]
    """
    prompt = build_prompt(user_query, registry)
    full_prompt = prompt + '{"name":"'
    working_ids = model.encode(full_prompt)[0].tolist()
    function_name: str = select_function_name(model, working_ids, registry)
    spec: FunctionSpec = registry[function_name]
    working_ids = model.encode(
        full_prompt + function_name + '", "parameters": {'
    )[0].tolist()
    parameters: dict[str, Any] = {}

    for i, (param_name, param_info) in enumerate(spec.parameters.items()):
        prefix: str = f'"{param_name}": '
        # 1. Update context with the parameter key prefix
        working_ids = model.encode(model.decode(working_ids) + prefix)[
            0
        ].tolist()

        param_type = param_info.get("type")
        if param_type == "boolean":
            value = select_boolean_value(model, working_ids)
            parameters[param_name] = value
            working_ids = model.encode(
                model.decode(working_ids) + ("true" if value else "false")
            )[0].tolist()

        elif param_type in ("integer", "number"):
            raw_value = select_number_value(model, working_ids)
            try:
                if param_type == "number":
                    parameters[param_name] = float(raw_value)
                else:
                    parameters[param_name] = int(float(raw_value))
            except ValueError:
                parameters[param_name] = 0 if param_type == "integer" else 0.0

            working_ids = model.encode(
                model.decode(working_ids) + raw_value
            )[0].tolist()

        elif param_type == "string":
            working_ids_with_quote = model.encode(
                model.decode(working_ids) + '"'
            )[0].tolist()
            raw_value = select_string_value(model, working_ids_with_quote)
            raw_value = raw_value.replace("\\\\", "\\").strip()
            parameters[param_name] = raw_value
            working_ids = model.encode(
                model.decode(working_ids) + f'"{raw_value}"'
            )[0].tolist()
        else:
            parameters[param_name] = None

        if i < len(spec.parameters) - 1:
            working_ids = model.encode(model.decode(working_ids) + ", ")[
                0
            ].tolist()

    return {
        "prompt": user_query,
        "name": function_name,
        "parameters": parameters,
    }
