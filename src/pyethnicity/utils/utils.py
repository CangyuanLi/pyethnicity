import math
from collections.abc import Sequence
from typing import SupportsFloat, SupportsIndex, Union

import polars as pl
import requests

from .paths import DAT_PATH
from .types import ArrayLike

IntoExpr = Union[str, pl.Expr]

RACES = ("asian", "black", "hispanic", "white")
RACES_6 = ("asian", "black", "hispanic", "multiple", "native", "white")


def _as_expr(e: IntoExpr) -> pl.Expr:
    if isinstance(e, str):
        return pl.col(e)
    elif isinstance(e, pl.Expr):
        return e
    else:
        raise ValueError("Invalid input.")


def _assert_equal_lengths(*inputs: Union[object, ArrayLike]):
    lengths = []

    for input in inputs:
        if not hasattr(input, "__len__") or isinstance(input, str):
            input = [input]

        lengths.append(len(input))

    mean_length = sum(lengths) / len(lengths)

    if any(length != mean_length for length in lengths):
        raise ValueError("All inputs need to be of equal length.")


def _remove_single_chars(expr: IntoExpr) -> pl.Expr:
    return (
        _as_expr(expr)
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_bytes() > 1))
        .list.join(" ")
    )


def _remove_generational_suffixes(expr: IntoExpr) -> pl.Expr:
    return (
        _as_expr(expr)
        .str.strip_suffix(" jr")
        .str.strip_suffix(" sr")
        .str.strip_suffix(" iii")
        .str.strip_suffix(" iv")
    )


def _std_norm(values: Sequence[float]) -> list[float]:
    total = sum(values)

    return [v / total for v in values]


def _is_null(x: Union[SupportsFloat, SupportsIndex]):
    return math.isnan(x) or x is None


def _set_name(x, name: str):
    try:
        name = x.name
    except AttributeError:
        pass

    return name


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


def _sum_horizontal(*exprs: IntoExpr) -> pl.Expr:
    exprs = [_as_expr(e) for e in exprs]

    return (
        pl.when(pl.all_horizontal(e.is_null() for e in exprs))
        .then(None)
        .otherwise(pl.sum_horizontal(exprs))
    )
