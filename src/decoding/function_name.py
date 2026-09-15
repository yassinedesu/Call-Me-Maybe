import math

from llm_sdk import Small_LLM_Model

from ..models import FunctionSpec


def select_function_name(
        model: Small_LLM_Model,
        input_ids: list[int],
        registry: dict[str, FunctionSpec],
) -> str:
    """
    Selects a function name based on a sequence of input tokens, predefined
    in a registry, using a given model for token generation. The function
    iteratively generates tokens by considering the logits from the model
    and enforcing constraints based on the provided function registry. The
    process stops once a valid function name or closure token is identified.

    :param model: An instance of `Small_LLM_Model` used for encoding,
        decoding, and generating logits for token prediction.
    :type model: Small_LLM_Model
    :param input_ids: A list of integer input token IDs representing the
        initial context for token generation.
    :type input_ids: list[int]
    :param registry: A dictionary mapping function names (keys) to their
        specifications (`FunctionSpec`). These names are used for determining
        the constraints during token generation.
    :type registry: dict[str, FunctionSpec]
    :return: The selected function name as a string, decoded from the
        generated token sequence.
    :rtype: str
    :raises ValueError: If the registry is empty, indicating that no function
        names are available for selection.
    """
    if not registry:
        raise ValueError("Function registry is empty; nothing to select from.")

    working_ids = list(input_ids)
    generated_tokens: list[int] = []
    valid_sequence = [model.encode(name)[0].tolist() for name in registry]
    closure_id = model.encode('"')[0].tolist()[-1]
    max_steps: int = max(len(seq) for seq in valid_sequence) + 1

    for _ in range(max_steps):
        logits = model.get_logits_from_input_ids(working_ids)

        allowed_tokens: set[int] = set()
        is_complete: bool = False

        for seq in valid_sequence:
            if seq[: len(generated_tokens)] == generated_tokens:
                if len(seq) > len(generated_tokens):
                    allowed_tokens.add(seq[len(generated_tokens)])
                else:
                    is_complete = True
        if is_complete:
            allowed_tokens.add(closure_id)

        masked: list[float] = [-math.inf] * len(logits)
        for tok_id in allowed_tokens:
            masked[tok_id] = logits[tok_id]
        next_token_id = masked.index(max(masked))

        generated_tokens.append(next_token_id)
        working_ids.append(next_token_id)
        if next_token_id == closure_id:
            break

    final_tokens = [t for t in generated_tokens if t != closure_id]
    return model.decode(final_tokens)
