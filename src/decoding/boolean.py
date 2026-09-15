import math

from llm_sdk import Small_LLM_Model


def select_boolean_value(
        model: Small_LLM_Model,
        input_ids: list[int],
) -> bool:
    """
    Determine the boolean value ("true" or "false") predicted by a small
    language model for given input IDs. The function evaluates the logits
    corresponding to "true" and "false" tokens, masking out all other logits.
    It then selects the token with the highest probability as the predicted
    boolean value.

    :param model: The small language model instance providing methods to
        process and encode input IDs.
    :type model: Small_LLM_Model
    :param input_ids: A list of integer input token IDs to be evaluated by
        the language model.
    :type input_ids: list[int]
    :return: A boolean value indicating the model's prediction (True for
        "true", False for "false").
    :rtype: bool
    """
    logits = model.get_logits_from_input_ids(input_ids)
    true_id = model.encode("true")[0].tolist()[-1]
    false_id = model.encode("false")[0].tolist()[-1]

    masked: list[float] = [-math.inf] * len(logits)
    masked[true_id] = logits[true_id]
    masked[false_id] = logits[false_id]

    return bool(masked.index(max(masked)) == true_id)
