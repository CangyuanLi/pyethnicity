import functools
import math
from collections.abc import Sequence
from typing import SupportsFloat, SupportsIndex, Union

import requests

from .paths import DAT_PATH
from .types import ArrayLike

RACES = ("asian", "black", "hispanic", "white")


def _assert_equal_lengths(*inputs: Union[object, ArrayLike]):
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


def _std_norm(values: Sequence[float]) -> list[float]:
    total = sum(values)

    return [v / total for v in values]


def _is_null(x: Union[SupportsFloat, SupportsIndex]):
    return math.isnan(x) or x is None


def _download(path: str):
    r = requests.get(
        f"https://raw.githubusercontent.com/CangyuanLi/pyethnicity/master/src/pyethnicity/data/{path}"
    )
    if r.status_code != 200:
        raise requests.exceptions.HTTPError(f"{r.status_code}: DOWNLOAD FAILED")

    parent_folder = path.split("/")[0]
    (DAT_PATH / parent_folder).mkdir(exist_ok=True)

    with open(DAT_PATH / path, "wb") as f:
        f.write(r.content)
