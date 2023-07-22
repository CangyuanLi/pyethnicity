import string

import pandas as pd
import polars as pl

from .utils.paths import DIST_PATH
from .utils.types import ArrayLike
from .utils.utils import _assert_equal_lengths, _remove_single_chars


class BayesianModel:
    def __init__(self):
        self._races = ("asian", "black", "hispanic", "white")
        self._unwanted_chars = string.digits + string.punctuation + string.whitespace

        self._PROB_RACE_GIVEN_FIRST_NAME = pl.read_parquet(
            DIST_PATH / "prob_race_given_first_name.parquet"
        )
        self._PROB_FIRST_NAME_GIVEN_RACE = pl.read_parquet(
            DIST_PATH / "prob_first_name_given_race.parquet"
        )
        self._PROB_RACE_GIVEN_LAST_NAME = pl.read_parquet(
            DIST_PATH / "prob_race_given_last_name.parquet"
        )
        self._PROB_ZCTA_GIVEN_RACE = pl.read_parquet(
            DIST_PATH / "prob_zcta_given_race_2010.parquet"
        )
        self._PROB_RACE_GIVEN_ZCTA = pl.read_parquet(
            DIST_PATH / "prob_race_given_zcta_2010.parquet"
        )

    def _remove_chars(self, expr: pl.Expr) -> pl.Expr:
        for char in self._unwanted_chars:
            expr = expr.str.replace_all(char, "", literal=True)

        return expr

    def _normalize_name(self, name: str | ArrayLike, col_name: str) -> pl.DataFrame:
        if isinstance(name, str):
            name = [name]

        return (
            pl.LazyFrame({col_name: name})
            .with_columns(
                pl.col(col_name)
                .pipe(self._remove_chars)
                .str.to_uppercase()
                .str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
                .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
                .str.replace_all(r"\s?III\s*?$", "")
                .str.replace_all(r"\s?IV\s*?$", "")
                .apply(_remove_single_chars)
            )
            .collect()
        )

    @staticmethod
    def _normalize_zcta(zcta: int | str | ArrayLike, col_name: str) -> pl.DataFrame:
        if isinstance(zcta, str):
            zcta = [zcta]

        return (
            pl.LazyFrame({col_name: zcta})
            .with_columns(pl.col(col_name).cast(str).str.zfill(5))
            .collect()
        )

    def _bng(
        self, prob_race_given_name: pl.DataFrame, zcta: int | str | ArrayLike
    ) -> pd.DataFrame:
        zcta = self._normalize_zcta(zcta, "zcta5")
        prob_zcta_given_race = zcta.join(
            self._PROB_ZCTA_GIVEN_RACE, on="zcta5", how="left"
        )

        numer = prob_race_given_name.select(self._races) * prob_zcta_given_race.select(
            self._races
        )
        denom = numer.sum(axis=1)
        probs = numer / denom

        return pl.concat(
            [prob_race_given_name.select("first_name", "last_name"), probs],
            how="horizontal",
        ).to_pandas()

    def bisg(
        self, last_name: str | ArrayLike, zcta: int | str | ArrayLike
    ) -> pd.DataFrame:
        _assert_equal_lengths(last_name, zcta)

        last_name_cleaned = self._normalize_name(last_name, "last_name")
        zcta_cleaned = self._normalize_zcta(zcta, "zcta5")

        df = pl.concat([last_name_cleaned, zcta_cleaned], how="horizontal")

        prob_race_given_last_name = df.join(
            self._PROB_RACE_GIVEN_LAST_NAME,
            left_on="last_name",
            right_on="name",
            how="left",
        ).select(self._races)

        prob_zcta_given_race = df.join(
            self._PROB_ZCTA_GIVEN_RACE, on="zcta5", how="left"
        ).select(self._races)

        bisg_numer = prob_race_given_last_name * prob_zcta_given_race
        bisg_denom = bisg_numer.sum(axis=1)
        bisg_probs = bisg_numer / bisg_denom

        df = pd.DataFrame()
        df["last_name"] = last_name
        df["zcta"] = zcta

        return pd.concat([df, bisg_probs.to_pandas()], axis=1)

    def bifsg(
        self,
        first_name: str | ArrayLike,
        last_name: str | ArrayLike,
        zcta: int | str | ArrayLike,
    ) -> pd.DataFrame:
        _assert_equal_lengths(first_name, last_name, zcta)

        first_name_cleaned = self._normalize_name(first_name, "first_name")
        last_name_cleaned = self._normalize_name(last_name, "last_name")
        zcta_cleaned = self._normalize_zcta(zcta, "zcta5")

        df = pl.concat(
            [first_name_cleaned, last_name_cleaned, zcta_cleaned], how="horizontal"
        )

        prob_first_name_given_race = df.join(
            self._PROB_FIRST_NAME_GIVEN_RACE,
            left_on="first_name",
            right_on="name",
            how="left",
        ).select(self._races)

        prob_race_given_last_name = df.join(
            self._PROB_RACE_GIVEN_LAST_NAME,
            left_on="last_name",
            right_on="name",
            how="left",
        ).select(self._races)

        prob_zcta_given_race = df.join(
            self._PROB_ZCTA_GIVEN_RACE, on="zcta5", how="left"
        ).select(self._races)

        bifsg_numer = (
            prob_first_name_given_race
            * prob_race_given_last_name
            * prob_zcta_given_race
        )
        bifsg_denom = bifsg_numer.sum(axis=1)
        bifsg_probs = bifsg_numer / bifsg_denom

        df = pd.DataFrame()
        df["first_name"] = first_name
        df["last_name"] = last_name
        df["zcta"] = zcta

        return pd.concat([df, bifsg_probs.to_pandas()], axis=1)
