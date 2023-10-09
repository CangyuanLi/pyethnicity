from __future__ import annotations

import string
from typing import Literal, Optional

import cutils
import numpy as np
import onnxruntime
import pandas as pd
import polars as pl
import tqdm

from ._bayesian_models import _bng, bifsg, bisg
from .utils.paths import MODEL_PATH
from .utils.types import Geography, GeoType, Model, Name
from .utils.utils import (
    RACES,
    _assert_equal_lengths,
    _is_null,
    _remove_single_chars,
    _std_norm,
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
        }

    def load(self, model: Model) -> onnxruntime.InferenceSession:
        if self._models[model] is None:
            self._models[model] = onnxruntime.InferenceSession(
                MODEL_PATH / f"{model}.onnx",
                providers=onnxruntime.get_available_providers(),
            )

        return self._models[model]


MODEL_LOADER = ModelLoader()


def _encode_name(name: str, mapper: dict = CHAR_MAPPER, max_len: int = 15):
    ids = [0] * max_len
    for i, c in enumerate(name):
        if i < max_len:
            ids[i] = mapper.get(c, mapper["U"])

    return ids


def _normalize_name(name: Name) -> list[str]:
    if isinstance(name, str):
        name = [name]

    return (
        pl.Series(values=name)
        .str.to_uppercase()
        .str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?III\s*?$", "")
        .str.replace_all(r"\s?IV\s*?$", "")
        .str.to_lowercase()
        .str.replace_all(f"[^{VALID_NAME_CHARS}]", "")
        .apply(_remove_single_chars)
    ).to_list()


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
    first_name: Name, last_name: Name, chunksize: int = CHUNKSIZE
) -> pd.DataFrame:
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
    pd.DataFrame
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
    _assert_equal_lengths(first_name, last_name)

    first_name_cleaned = _normalize_name(first_name)
    last_name_cleaned = _normalize_name(last_name)

    X = [
        _encode_name(fn) + _encode_name(ln)
        for fn, ln in zip(first_name_cleaned, last_name_cleaned)
    ]
    X = _pad_sequences(X, maxlen=30).astype(np.float32)

    model = MODEL_LOADER.load("first_last")

    input_name = model.get_inputs()[0].name

    y_pred = []
    for input_ in tqdm.tqdm(cutils.chunk_seq(X, chunksize)):
        y_pred.extend(model.run(None, input_feed={input_name: input_})[0])

    preds: dict[str, list] = {r: [] for r in RACES}
    for row in y_pred:
        for idx, p in enumerate(row):
            preds[RACE_MAPPER[idx]].append(p)

    df = pd.DataFrame()
    for r, v in preds.items():
        df[r] = v
    df.insert(0, "first_name", first_name)
    df.insert(1, "last_name", last_name)

    return df


def predict_race_flg(
    first_name: Name,
    last_name: Name,
    geography: Geography,
    geo_type: GeoType,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
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
    pd.DataFrame
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
    fl_preds = predict_race_fl(first_name, last_name, chunksize)

    return _bng(pl.from_pandas(fl_preds), geography, geo_type)


def predict_race(
    first_name: Name,
    last_name: Name,
    geography: Geography,
    geo_type: GeoType,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
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
    pd.DataFrame
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
        - data/distributionsprob_first_name_given_race.parquet

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
    flz = predict_race_flg(first_name, last_name, geography, geo_type, chunksize)
    bifsg_ = bifsg(first_name, last_name, geography, geo_type)
    bisg_ = bisg(last_name, geography, geo_type)

    weights = [1, 1, 1]

    res: dict[str, list] = {r: [] for r in RACES}
    for race in RACES:
        for row in zip(
            flz[race].to_list(), bifsg_[race].to_list(), bisg_[race].to_list()
        ):
            valid_inputs = []
            valid_weights = []
            for r, w in zip(row, weights):
                if not _is_null(r):
                    valid_inputs.append(r)
                    valid_weights.append(w)

            valid_weights = _std_norm(valid_weights)

            res[race].append(sum(i * w for i, w in zip(valid_inputs, valid_weights)))

    df = pd.DataFrame()
    for r, v in res.items():
        df[r] = v
    df.insert(0, "first_name", first_name)
    df.insert(1, "last_name", last_name)
    df.insert(2, geo_type, geography)

    return df
