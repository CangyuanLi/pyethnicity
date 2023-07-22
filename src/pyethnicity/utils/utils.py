import functools

from utils.types import ArrayLike


def _assert_equal_lengths(*inputs: object | ArrayLike):
    lengths = []

    for input in inputs:
        if not isinstance(input, ArrayLike) or isinstance(input, str):
            input = [input]

        lengths.append(len(input))

    mean_length = sum(lengths) / len(lengths)

    if any(length != mean_length for length in lengths):
        raise ValueError("All inputs need to be of equal length.")


@functools.lru_cache()
def _remove_single_chars(name: str) -> str:
    return " ".join(part for part in name.split(" ") if len(part) > 1)