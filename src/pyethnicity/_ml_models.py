from __future__ import annotations

import concurrent.futures
import multiprocessing
import string
from typing import Literal, Optional

import cutils
import numpy as np
import onnxruntime
import polars as pl
import polars.selectors as cs
import tqdm
from polars.type_aliases import IntoExpr

from ._bayesian_models import _bng, bifsg, bisg
from .utils.paths import MODEL_PATH
from .utils.types import Geography, Model, Name
from .utils.utils import (
    RACES,
    _as_expr,
    _download,
    _remove_generational_suffixes,
    _remove_single_chars,
    _set_name,
    _sum_horizontal,
)

VALID_NAME_CHARS = f"{string.ascii_lowercase} '-"
VALID_NAME_CHARS_DICT = {c: i for i, c in enumerate(VALID_NAME_CHARS, start=1)}
CHAR_MAPPER = {c: i for i, c in enumerate(f"{string.ascii_lowercase} U", start=1)}
RACE_MAPPER = {i: r for i, r in enumerate(RACES)}
CHUNKSIZE = 1028


class ModelLoader:
    def __init__(self):
        self._models: dict[Model] = {
            "first_last": None,
            "first_sex": None,
        }

    def load(self, model: Model) -> onnxruntime.InferenceSession:
        if self._models[model] is None:
            file = f"{model}.onnx"
            if not (MODEL_PATH / file).exists():
                _download(f"models/{file}")

            self._models[model] = onnxruntime.InferenceSession(
                MODEL_PATH / file,
                providers=onnxruntime.get_available_providers(),
            )

        return self._models[model]


MODEL_LOADER = ModelLoader()


def _encode_name(name: str, mapper: dict = CHAR_MAPPER, max_len: int = 15) -> list[int]:
    out_of_vocab = mapper["U"] if mapper == CHAR_MAPPER else ""
    ids = [0] * max_len
    for i, c in enumerate(name):
        if i < max_len:
            ids[i] = mapper.get(c, out_of_vocab)

    return ids


def _encode_full_name(name_tuple: tuple[str, str]) -> list[int]:
    first_name, last_name = name_tuple

    return _encode_name(first_name) + _encode_name(last_name)


def _normalize_name(expr: IntoExpr) -> pl.Expr:
    return (
        _as_expr(expr)
        .str.to_lowercase()
        .pipe(_remove_generational_suffixes)
        .str.replace_all(f"[^{VALID_NAME_CHARS}]", "")
        .pipe(_remove_single_chars)
    )


# this is taken from keras code, this is just to avoid having to import the entirety
# of tensorflow just for this function
def _pad_sequences(
    sequences,
    maxlen: Optional[int] = None,
    dtype="int32",
    padding: Literal["pre", "post"] = "pre",
    truncating: Literal["pre", "post"] = "pre",
    value: float = 0.0,
) -> np.ndarray:
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                f"`sequences` must be a list of iterables. Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')

    return x


def predict_race_fl(
    first_name: Name,
    last_name: Name,
    chunksize: int = CHUNKSIZE,
    _model: onnxruntime.InferenceSession = None,
) -> pl.DataFrame:
    """Predict race from first and last name.

    Parameters
    ----------
    first_name : Name
        A string or array-like of strings
    last_name : Name
        A string or array-like of strings
    chunksize : int, optional
        How many rows are passed to the ONNX session at a time, by default 1028

    Returns
    -------
    pl.DataFrame
        A DataFrame of first_name, last_name, and `P(r|f,s)` for Asian, Black,
        Hispanic, and White.

    Notes
    -----
    The data files can be found in:
        - data/models/first_last.onnx

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.predict_race_fl(first_name="cangyuan", last_name="li")
    >>> pyethnicity.predict_race_fl(
            first_name=["cangyuan", "mark"], last_name=["li", "luo"]
        )
    """
    df = (
        pl.LazyFrame({"first_name": first_name, "last_name": last_name})
        .unique()
        .with_columns(
            pl.col("first_name", "last_name")
            .pipe(_normalize_name)
            .name.map(lambda c: f"{c}_cleaned")
        )
        .collect()
    )

    first_name_cleaned = df["first_name_cleaned"]
    last_name_cleaned = df["last_name_cleaned"]
    X = []
    for tup in zip(first_name_cleaned, last_name_cleaned):
        X.append(_encode_full_name(tup))

    # with concurrent.futures.ProcessPoolExecutor(
    #     mp_context=multiprocessing.get_context("fork")
    # ) as pool:
    #     X = list(
    #         pool.map(_encode_full_name, zip(first_name_cleaned, last_name_cleaned))
    #     )

    # print(X)

    X = _pad_sequences(X, maxlen=30).astype(np.float32)

    if _model is None:
        _model = MODEL_LOADER.load("first_last")

    input_name = _model.get_inputs()[0].name

    with tqdm.tqdm(total=len(X)) as pbar:
        y_pred = []
        for input_ in cutils.chunk_seq(X, chunksize):
            y_pred.extend(_model.run(None, input_feed={input_name: input_})[0])
            pbar.update(len(input_))

    preds: dict[str, list] = {r: [] for r in RACES}
    for row in y_pred:
        for idx, p in enumerate(row):
            preds[RACE_MAPPER[idx]].append(p)

    first_name_col = _set_name(first_name, "first_name")
    last_name_col = _set_name(last_name, "last_name")

    res = pl.concat(
        [
            df.select("first_name", "last_name").rename(
                {"first_name": first_name_col, "last_name": last_name_col}
            ),
            pl.DataFrame(preds),
        ],
        how="horizontal",
    )

    return res


def predict_race_flg(
    first_name: Name,
    last_name: Name,
    zcta: Optional[Geography] = None,
    tract: Optional[Geography] = None,
    block_group: Optional[Geography] = None,
    chunksize: int = CHUNKSIZE,
    _model: Optional[onnxruntime.InferenceSession] = None,
) -> pl.DataFrame:
    r"""Predict race from first name, last name, and geography. The output from
    pyethnicity.predict_race_fl is combined with geography using Naive Bayes:

    .. math::

        P(r|n,g) = \frac{P(r|n) \times P(g|r)}{\sum_{r=1}^4 P(r|n) \times P(g|r)}

    where `r` is race, `n` is name, and `g` is geography. The sum is across all races,
    i.e. Asian, Black, Hispanic, and White.

    Parameters
    ----------
    first_name : Name
        A string or array-like of strings
    last_name : Name
        A string or array-like of strings
    geography : Geography
        A scalar or array-like of geographies
    geo_type : GeoType
         One of `zcta` or `tract`
    chunksize : int, optional
        How many rows are passed to the ONNX session at a time, by default 1028

    Returns
    -------
    pl.DataFrame
        A DataFrame of first_name, last_name, geography, and `P(r|n,g)` for Asian,
        Black, Hispanic, and White.

    Notes
    -----
    The data files can be found in:
        - data/models/first_last.onnx
        - data/distributions/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2010.parquet
        - data/distributions/prob_tract_given_race_2010.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.predict_race_flg(
    >>>     first_name="cangyuan", last_name="li", geography=11106, geo_type="zcta"
    >>> )
    >>> pyethnicity.predict_race_flg(
    >>>     first_name=["cangyuan", "mark"], last_name=["li", "luo"],
    >>>     geography=[11106, 27106], geo_type="zcta"
    >>> )
    """
    fl_preds = predict_race_fl(
        first_name=first_name, last_name=last_name, chunksize=chunksize, _model=_model
    )

    first_name_col = _set_name(first_name, "first_name")
    last_name_col = _set_name(last_name, "last_name")
    zcta_col = _set_name(zcta, "zcta")
    tract_col = _set_name(tract, "tract")
    block_group_col = _set_name(block_group, "block_group")

    valid = pl.DataFrame(
        {
            first_name_col: first_name,
            last_name_col: last_name,
            zcta_col: zcta,
            tract_col: tract,
            block_group_col: block_group,
        }
    ).join(
        fl_preds.select(first_name_col, last_name_col),
        on=[first_name_col, last_name_col],
        how="inner",
    )

    zcta = valid[zcta_col] if zcta is not None else None
    tract = valid[tract_col] if tract is not None else None
    block_group = valid[block_group_col] if block_group is not None else None

    return _bng(fl_preds, zcta=zcta, tract=tract, block_group=block_group)


def predict_race(
    first_name: Name,
    last_name: Name,
    zcta: Optional[Geography] = None,
    tract: Optional[Geography] = None,
    block_group: Optional[Geography] = None,
    weights: list[float] = [1, 1, 1],
    chunksize: int = CHUNKSIZE,
    _model: onnxruntime.InferenceSession = None,
) -> pl.DataFrame:
    """Predict race from first name, last name, and geography. The output from
    pyethnicity.predict_race_flg is ensembled with pyethnicty.bisg and pyethnicty.bifsg.

    Parameters
    ----------
    first_name : Name
        A string or array-like of strings
    last_name : Name
        A string or array-like of strings
    geography : Geography
        A scalar or array-like of geographies
    geo_type : GeoType
         One of `zcta` or `tract`
    chunksize : int, optional
        How many rows are passed to the ONNX session at a time, by default 1028

    Returns
    -------
    pl.DataFrame
        A DataFrame of first_name, last_name, geography, and `P(r|n,g)` for Asian,
        Black, Hispanic, and White. If the geography cannot be found, the probability
        is `NaN`.

    Notes
    -----
    The data files can be found in:
        - data/models/first_last.onnx
        - data/distributions/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2010.parquet
        - data/distributions/prob_tract_given_race_2010.parquet
        - data/distributions/prob_first_name_given_race.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.predict_race(
    >>>     first_name="cangyuan", last_name="li", geography=11106, geo_type="zcta"
    >>> )
    >>> pyethnicity.predict_race(
    >>>     first_name=["cangyuan", "mark"], last_name=["li", "luo"],
    >>>     geography=[11106, 27106], geo_type="zcta"
    >>> )
    """
    name_mapper = {
        "first_name": _set_name(first_name, "first_name"),
        "last_name": _set_name(last_name, "last_name"),
        "zcta": _set_name(zcta, "zcta"),
        "tract": _set_name(tract, "tract"),
        "block_group": _set_name(block_group, "block_group"),
    }

    data = {"zcta": zcta, "tract": tract, "block_group": block_group}
    geo_cols = [name_mapper[k] for k, v in data.items() if v is not None]

    flg = predict_race_flg(
        first_name=first_name,
        last_name=last_name,
        zcta=zcta,
        tract=tract,
        block_group=block_group,
        chunksize=chunksize,
        _model=_model,
    )
    bifsg_ = bifsg(
        first_name=first_name,
        last_name=last_name,
        zcta=zcta,
        tract=tract,
        block_group=block_group,
        drop_intermediate=True,
    )
    bisg_ = bisg(
        last_name=last_name,
        zcta=zcta,
        tract=tract,
        block_group=block_group,
        drop_intermediate=True,
    )

    flg_weight, bifsg_weight, bisg_weight = weights

    return (
        flg.join(
            bifsg_,
            on=[name_mapper["first_name"], name_mapper["last_name"], *geo_cols],
            how="left",
            validate="m:1",
            suffix="_bifsg",
            coalesce=True,
        )
        .join(
            bisg_,
            on=[name_mapper["last_name"], *geo_cols],
            how="left",
            validate="m:1",
            suffix="_bisg",
            coalesce=True,
        )
        .with_columns(
            pl.lit(flg_weight).alias("flg_weight"),
            pl.lit(bifsg_weight).alias("bifsg_weight"),
            pl.lit(bisg_weight).alias("bisg_weight"),
        )
        .with_columns(
            pl.col("flg_weight") * pl.col("asian").is_not_null(),
            pl.col("bifsg_weight") * pl.col("asian_bifsg").is_not_null(),
            pl.col("bisg_weight") * pl.col("asian_bisg").is_not_null(),
        )
        .with_columns(
            _sum_horizontal("flg_weight", "bifsg_weight", "bisg_weight").alias(
                "total_weight"
            )
        )
        .with_columns(
            pl.col("flg_weight", "bifsg_weight", "bisg_weight").truediv(
                pl.col("total_weight")
            )
        )
        .with_columns(
            _sum_horizontal(
                pl.col(r) * pl.col("flg_weight"),
                pl.col(f"{r}_bifsg") * pl.col("bifsg_weight"),
                pl.col(f"{r}_bisg") * pl.col("bisg_weight"),
            ).alias(r)
            for r in RACES
        )
        .select(name_mapper["first_name"], name_mapper["last_name"], *geo_cols, *RACES)
    )


def predict_sex_f(
    first_name: Name,
    chunksize: int = CHUNKSIZE,
    _model: Optional[onnxruntime.InferenceSession] = None,
) -> pl.DataFrame:
    """Predict sex from first name.

    Parameters
    ----------
    first_name : Name
        A string or array-like of strings
    chunksize : int, optional
        How many rows are passed to the ONNX session at a time, by default 1028

    Returns
    -------
    pl.DataFrame
        A DataFrame of first_name, pct_male, and pct_female.

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.predict_sex_f(first_name="cangyuan")
    >>> pyethnicity.predict_sex_f(first_name=["cangyuan", "mercy"])
    """
    first_name_cleaned = _normalize_name(first_name)

    X = [_encode_name(fn, mapper=VALID_NAME_CHARS_DICT) for fn in first_name_cleaned]
    X = _pad_sequences(X, maxlen=15).astype(np.float32)

    if _model is None:
        _model = MODEL_LOADER.load("first_sex")

    input_name = _model.get_inputs()[0].name

    with tqdm.tqdm(total=len(X)) as pbar:
        y_pred = []
        for input_ in cutils.chunk_seq(X, chunksize):
            y_pred.extend(_model.run(None, input_feed={input_name: input_})[0])
            pbar.update(len(input_))

    pct_male = [row[0] for row in y_pred]

    return pl.DataFrame(
        {_set_name(first_name, "first_name"): first_name, "male": pct_male}
    ).with_columns(female=1 - pl.col("male"))
