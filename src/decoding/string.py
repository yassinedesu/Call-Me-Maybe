import json
import math

from llm_sdk import Small_LLM_Model


def _load_structural_drift_tokens(model: Small_LLM_Model) -> set[int]:
    """
    Loads a set of structural drift tokens from the model's vocabulary file.
    Structural drift tokens are identified based on the presence of certain
    marker characters in the decoded strings of token IDs.

    :param model: The model instance from which the path to the vocabulary
        file is derived and tokens are decoded.
    :type model: Small_LLM_Model
    :return: A set of token IDs that are considered structural drift tokens
        based on the defined markers.
    :rtype: set[int]
    """
    path = model.get_path_to_vocab_file()
    our_markers: set[str] = {
        "'",
        "regex",
        "replacement",
        "source_string",
        "{",
        "}",
        ".",
        "\\",
        "\n",
    }
    with open(path, "r") as f:
        vocab = json.load(f)

    banned: set[int] = set()
    for token_id in vocab.values():
        decoded_str = model.decode([token_id])
        if any(c in decoded_str for c in our_markers):
            banned.add(token_id)

    return banned


def _load_banned_string_tokens(model: Small_LLM_Model) -> set[int]:
    """
    Loads and returns a set of banned string tokens for a given model.

    This function processes the vocabulary file of the provided language model
    to identify and return token IDs that correspond to strings containing
    forbidden characters or substrings. Forbidden strings are defined in the
    `forbidden_badwords` set within the function.

    :param model: The language model whose vocabulary file will be processed.
    :type model: Small_LLM_Model

    :return: A set of token IDs corresponding to forbidden strings.
    :rtype: set[int]
    """
    banned: set[int] = set()
    forbidden_badwords: set[str] = {'"', "\n", "\r", "”", "“", "‘", "’"}
    path = model.get_path_to_vocab_file()
    with open(path, "r") as fd:
        vocab = json.load(fd)

    for token_id in vocab.values():
        decoded_str = model.decode([token_id])
        if any(char in decoded_str for char in forbidden_badwords):
            banned.add(token_id)
    return banned


def _would_repeat(
        tokens: list[int],
        next_id: int,
        min_period: int = 1,
        max_period: int = 6,
) -> bool:
    """
    Determines whether adding a new token to a sequence of tokens would lead
    to a repeated sequence (within a specified range of periods).

    The function checks for repetition by simulating the addition of `next_id`
    to the `tokens` sequence and analyzing segments of the list for repeated
    patterns with periods ranging from `min_period` to `max_period`. If
    repetition is detected, the function returns True; otherwise, it returns
    False.

    :param tokens: The list of existing tokens to evaluate for potential
        repetition.
    :param next_id: The next token to be added to the `tokens` list.
    :param min_period: The minimum period to consider for repetition analysis.
    :param max_period: The maximum period to consider for repetition analysis.
    :return: Boolean indicating whether adding the `next_id` results in
        repetition.
    """
    hypothetical_list = tokens + [next_id]
    for period in range(min_period, max_period + 1):
        if len(hypothetical_list) < 2 * period:
            continue
        the_stuff = hypothetical_list[-period:]
        before_it = hypothetical_list[-2 * period:-period]
        if the_stuff == before_it:
            return True
    return False


def select_string_value(
        model: Small_LLM_Model,
        input_ids: list[int],
        max_tokens: int = 30,
) -> str:
    """
    Selects a string value from a language model based on input tokens and
    constraints.

    This function interacts with a small language model to generate a sequence
    of tokens, starting from a given input token list. It applies constraints
    such as excluding banned string tokens and handling structural drift tokens
    to generate a meaningful and restricted string output. The generation
    process halts under specific conditions, such as reaching a maximum token
    limit, repeating tokens, or encountering a closing quote.

    :param model: The language model instance used for token generation and
        encoding.
    :type model: Small_LLM_Model
    :param input_ids: A list of token IDs serving as the initial input to the
        model's generation process.
    :type input_ids: list[int]
    :param max_tokens: The maximum number of tokens to generate. Defaults to
        30.
    :type max_tokens: int, optional
    :return: The generated string decoded from the token sequence.
    :rtype: str
    """
    close_hammer = model.encode('"')[0].tolist()[-1]
    banned_strings: set[int] = _load_banned_string_tokens(model)
    banned_drifts: set[int] = _load_structural_drift_tokens(model)

    banned_strings.discard(close_hammer)
    banned_drifts.discard(close_hammer)
    working_ids = list(input_ids)
    generated_tokens: list[int] = []

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(working_ids)
        masked_enemy = list(logits)

        for tok_id in banned_strings:
            masked_enemy[tok_id] = -math.inf

        next_token = masked_enemy.index(max(masked_enemy))
        if next_token == close_hammer and len(generated_tokens) > 0:
            break
        if _would_repeat(generated_tokens, next_token):
            break
        if next_token in banned_drifts and len(generated_tokens) > 0:
            break

        working_ids.append(next_token)
        generated_tokens.append(next_token)

    return model.decode(generated_tokens)
