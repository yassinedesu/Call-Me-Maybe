import math

from llm_sdk import Small_LLM_Model


def select_number_value(
        model: Small_LLM_Model,
        input_ids: list[int],
        max_digits: int = 20,
) -> str:
    """
    Extracts a numeric or number-like value as a string from a model's
    sequential token generation process. The function ensures that only valid
    numeric characters, such as digits, a minus sign, and a decimal point,
    are allowed during the generation. Optionally, the process may terminate
    early if specific stopping characters (such as `}` or `,`) are generated,
    or if the specified `max_digits` limit is reached.

    :param model: A Small_LLM_Model object that provides token encoding,
        decoding, and logits generation functionality.
    :param input_ids: List of integer token IDs representing the initial
        input sequence.
    :param max_digits: Maximum number of additional tokens to decode for
        generating the number. Defaults to 20.
    :return: A decoded string representing the generated numeric or
        number-like value.
    """
    working_ids = list(input_ids)
    generated_tokens: list[int] = []
    allowed_ids: set[int] = set()
    stop_ids: set[int] = set()

    for char in "0123456789-.":
        allowed_ids.add(model.encode(char)[0].tolist()[-1])
    for char in "},":
        stop_ids.add(model.encode(char)[0].tolist()[-1])

    allowed_tokens = allowed_ids | stop_ids
    for _ in range(max_digits):
        logits = model.get_logits_from_input_ids(working_ids)
        masked: list[float] = [-math.inf] * len(logits)

        for tok_id in allowed_tokens:
            masked[tok_id] = logits[tok_id]

        next_id = masked.index(max(masked))
        if next_id in stop_ids:
            break
        generated_tokens.append(next_id)
        working_ids.append(next_id)

    return model.decode(generated_tokens)
