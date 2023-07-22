# ruff: noqa: E402

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import string

import keras
import pandas as pd
import polars as pl
import tensorflow as tf
from bayesian_models import BayesianModel
from utils.paths import MODEL_PATH
from utils.types import ArrayLike, Model
from utils.utils import _assert_equal_lengths, _remove_single_chars


class ModelLoader:
    def __init__(self):
        self._models: dict[Model] = {
            "bilstm": None,
            "dual_bilstm": None,
        }

    def load(self, model: Model):
        if self._models[model] is None:
            self._models[model] = keras.models.load_model(MODEL_PATH / f"{model}.h5")

        return self._models[model]


LOADER = ModelLoader()

VALID_NAME_CHARS = f"{string.ascii_lowercase} '-"
VALID_NAME_CHARS_DICT = {c: i for i, c in enumerate(VALID_NAME_CHARS, start=1)}
CHAR_MAPPER = {c: i for i, c in enumerate(f"{string.ascii_lowercase} U", start=1)}
RACES = ("asian", "black", "hispanic", "white")
RACE_MAPPER = {i: r for i, r in enumerate(RACES)}
BAYESIAN_MODEL = BayesianModel()


def _encode_name(name: str, mapper: dict = CHAR_MAPPER, max_len: int = 15):
    ids = [0] * max_len
    for i, c in enumerate(name):
        if i < max_len:
            ids[i] = mapper[c]

    return ids


def _normalize_name(name: str | ArrayLike) -> list[str]:
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


def predict_race_fl(
    first_name: str | ArrayLike, last_name: str | ArrayLike, backend: Model = "bilstm"
) -> pd.DataFrame:
    _assert_equal_lengths(first_name, last_name)

    first_name = _normalize_name(first_name)
    last_name = _normalize_name(last_name)

    if backend == "bilstm":
        X = [
            _encode_name(fn) + _encode_name(ln) for fn, ln in zip(first_name, last_name)
        ]
        X = tf.keras.utils.pad_sequences(X, maxlen=30)

        model = LOADER.load("bilstm")
    elif backend == "dual_bilstm":
        raise NotImplementedError("Dual BiLSTM is not yet implemented.")

    y_pred = model.predict(X, verbose=1)

    preds = {r: [] for r in RACES}
    for row in y_pred:
        for idx, p in enumerate(row):
            preds[RACE_MAPPER[idx]].append(p)

    res = pd.DataFrame(preds)
    res["first_name"] = first_name
    res["last_name"] = last_name

    return res


def predict_race_flg(
    first_name: str | ArrayLike,
    last_name: str | ArrayLike,
    zcta: int | str | ArrayLike,
    backend: Model = "bilstm",
) -> pd.DataFrame:
    fl_preds = predict_race_fl(first_name, last_name, backend)

    return BAYESIAN_MODEL._bng(pl.from_pandas(fl_preds), zcta)
