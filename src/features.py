import numpy as np


def _lowlevel_features(
    token: str, prefix: str | int | None = None, include_bias: bool = True
) -> dict[str, float | str | bool]:
    """Extracts low-level features from the token.

    Parameters
    ----------
    token : str
        The smallest unit to consider in a sentence. In
        this application it is considered to be a word, or
        a puntuation mark.
    prefix : str, int, optional
        Prefix to be placed before keynames in dictionary.
    include_bias : bool, default=True
        Whether to include bias into the feature dictionary.

    Returns
    -------
    dict[str, float | str | bool]
        Dictionary of features.
    """
    _features = {
        "token.lower()": token.lower(),
        "token[-3:]": token[-3:],
        "token[-2:]": token[-2:],
        "token[:3]": token[:3],
        "token[:2]": token[:2],
        "token.isupper()": token.isupper(),
        "token.istitle()": token.istitle(),
        "token.isdigit()": token.isdigit(),
        "contains_digit": any(char.isdigit() for char in token),
    }

    if include_bias:
        _features["bias"] = 1.0

    return (
        {f"{prefix}:{key}": value for key, value in _features.items()}
        if prefix is not None
        else _features
    )


def _token2features(tokens: list[str], token_index: int, window_size: int) -> dict:
    """Creates a feature dictionary for a single token.

    Parameters
    ----------
    tokens : list[str]
        Tokens of a sentence.
    token_index : int
        Index of the token among passed tokens.
    window_size : int
        Size of a sliding window (how many words before and after to include
        into a feature space).

    Returns
    -------
    dict
        Features in form of key:value
    """
    _features = {"BOS": token_index == 0, "EOS": token_index == len(tokens) - 1}

    for order_index, window_token in enumerate(
        tokens[
            np.clip(token_index - window_size, 0, len(tokens)) : np.clip(
                token_index + window_size, 0, len(tokens)
            )
        ],
        start=np.clip(token_index - window_size, 0, len(tokens)),
    ):
        if order_index == token_index:
            _features.update(_lowlevel_features(token=window_token))
        else:
            _features.update(
                _lowlevel_features(
                    token=window_token, prefix=order_index, include_bias=False
                )
            )

    return _features


def tokens2features(tokens: list[str], window_size: int = 2) -> list[dict]:
    """Translates each token to the corresponding space of features.

    Parameters
    ----------
    tokens : list[str]
        Tokens of a sentence.
    window_size : int, default=2
        Size of a sliding window (how many words before and after to include
        into a feature space).

    Returns
    -------
    list[dict]
        List of features.
    """
    return [
        _token2features(tokens=tokens, token_index=token_index, window_size=window_size)
        for token_index in range(len(tokens))
    ]


def numbers2labels(numbers: list[int]) -> list[str]:
    """Translates ids of the classes into corresponding
    names.

    Parameters
    ----------
    numbers : list[int]
        List of indices where:
        - 0: NON-ENTITY
        - 1: LOCATION
        - 2: PERSON
        - 4: MISC

    Returns
    -------
    list[str]
        Named labels.
    """
    result = []

    _map = {0: "NON-ENTITY", 1: "LOCATION", 2: "PERSON", 4: "MISC"}

    last_number = None
    for number in numbers:
        if number == 0:
            last_number = None
            result.append(_map[number])
        else:
            if last_number != number:
                last_number = number
                result.append(f"B-{_map[number]}")
            else:
                result.append(f"I-{_map[number]}")

    return result
