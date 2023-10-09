from __future__ import annotations

import string
import typing
from typing import Literal, Optional

import pandas as pd
import polars as pl

from .utils.paths import DIST_PATH
from .utils.types import Geography, GeoType, Name
from .utils.utils import RACES, _assert_equal_lengths, _remove_single_chars

UNWANTED_CHARS = string.digits + string.punctuation + string.whitespace

Resource = Literal[
    "prob_race_given_first_name",
    "prob_first_name_given_race",
    "prob_race_given_last_name",
    "prob_zcta_given_race_2010",
    "prob_race_given_zcta_2010",
    "prob_tract_given_race_2010",
    "prob_race_given_tract_2010",
]


class ResourceLoader:
    def __init__(self):
        self._resources: dict[Resource, Optional[pl.DataFrame]] = {
            k: None for k in typing.get_args(Resource)
        }

    def load(self, resource: Resource) -> pl.DataFrame:
        if self._resources[resource] is None:
            self._resources[resource] = pl.read_parquet(
                DIST_PATH / f"{resource}.parquet"
            )

        data = self._resources[resource]
        assert data is not None

        return data


RESOURCE_LOADER = ResourceLoader()


def _remove_chars(expr: pl.Expr) -> pl.Expr:
    for char in UNWANTED_CHARS:
        expr = expr.str.replace_all(char, "", literal=True)

    return expr


def _normalize_name(name: Name, col_name: str) -> pl.DataFrame:
    if isinstance(name, str):
        name = [name]

    return (
        pl.LazyFrame({col_name: name})
        .with_columns(
            pl.col(col_name)
            .pipe(_remove_chars)
            .str.to_uppercase()
            .str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
            .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
            .str.replace_all(r"\s?III\s*?$", "")
            .str.replace_all(r"\s?IV\s*?$", "")
            .apply(_remove_single_chars)
        )
        .collect()
    )


def _normalize_zcta(zcta: Geography, col_name: str = "zcta5") -> pl.DataFrame:
    if isinstance(zcta, str):
        zcta = [zcta]

    return (
        pl.LazyFrame({col_name: zcta})
        .with_columns(pl.col(col_name).cast(str).str.zfill(5))
        .collect()
    )


def _normalize_tract(tract: Geography, col_name: str = "tract") -> pl.DataFrame:
    if isinstance(tract, str):
        tract = [tract]

    return (
        pl.LazyFrame({col_name: tract})
        .with_columns(pl.col(col_name).cast(str).str.zfill(11))
        .collect()
    )


def _resolve_geography(geography: Geography, geo_type: GeoType) -> pl.DataFrame:
    if geo_type == "tract":
        geo = _normalize_tract(geography)
        prob_geo_given_race = geo.join(
            RESOURCE_LOADER.load("prob_tract_given_race_2010"), on="tract", how="left"
        )
    elif geo_type == "zcta":
        geo = _normalize_zcta(geography)
        prob_geo_given_race = geo.join(
            RESOURCE_LOADER.load("prob_zcta_given_race_2010"), on="zcta5", how="left"
        )
    else:
        raise ValueError(f"`{geo_type}` is not a valid geography.")

    return prob_geo_given_race


def _bng(
    prob_race_given_name: pl.DataFrame, geography: Geography, geo_type: GeoType
) -> pd.DataFrame:
    prob_geo_given_race = _resolve_geography(geography, geo_type)

    numer = prob_race_given_name.select(RACES) * prob_geo_given_race.select(RACES)
    denom = numer.sum(axis=1)
    probs = numer / denom

    return pl.concat(
        [prob_race_given_name.select("first_name", "last_name"), probs],
        how="horizontal",
    ).to_pandas()


def bisg(last_name: Name, geography: Geography, geo_type: GeoType) -> pd.DataFrame:
    r"""Implements Bayesian Improved Surname Geocoding (BISG), developed by
    Elliot et. al (2009) https://link.springer.com/article/10.1007/s10742-009-0047-1.
    Pyethnicity augments the Census surname list with distributions calculated from
    voter registration data sourced from L2.

    .. math::

        P(r|s,g) = \frac{P(r|s) \times P(g|r)}{\sum_{r=1}^4 P(r|s) \times P(g|r)}

    where `r` is race, `s` is surname, and `g` is geography. The sum is across all
    races, i.e. Asian, Black, Hispanic, and White.

    Parameters
    ----------
    last_name : Name
        A string or array-like of strings
    geography : Geography
        A scalar or array-like of geographies
    geo_type : GeoType
        One of `zcta` or `tract`

    Returns
    -------
    pd.DataFrame
        A DataFrame of last_name, geography, and `P(r | s, g)` for Asian, Black,
        Hispanic, and White. If either the last name or geography cannot be found,
        the probability is `NaN`.

    Notes
    -----
    The data files can be found in:
        - data/distributionsprob_race_given_last_name.parquet
        - data/distributionsprob_zcta_given_race_2010.parquet
        - data/distributionsprob_tract_given_race_2010.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.bisg(last_name="li", zcta=27106, geo_type="zcta")
    >>> pyethnicity.bisg(last_name=["li", "luo"], zcta=[27106, 11106], geo_type="zcta")
    """
    _assert_equal_lengths(last_name, geography)

    last_name_cleaned = _normalize_name(last_name, "last_name")

    prob_race_given_last_name = last_name_cleaned.join(
        RESOURCE_LOADER.load("prob_race_given_last_name"),
        left_on="last_name",
        right_on="name",
        how="left",
    ).select(RACES)

    prob_geo_given_race = _resolve_geography(geography, geo_type).select(RACES)

    bisg_numer = prob_race_given_last_name * prob_geo_given_race
    bisg_denom = bisg_numer.sum(axis=1)
    bisg_probs = bisg_numer / bisg_denom

    df = bisg_probs.to_pandas()
    df.insert(0, "last_name", last_name)
    df.insert(1, geo_type, geography)

    return df


def bifsg(
    first_name: Name, last_name: Name, geography: Geography, geo_type: GeoType
) -> pd.DataFrame:
    r"""Implements Bayesian Improved Firstname Surname Geocoding (BIFSG), developed by
    Voicu (2018) https://www.tandfonline.com/doi/full/10.1080/2330443X.2018.1427012.
    Pyethnicity augments the Census surname list and HMDA first name list with
    distributions calculated from voter registration data sourced from L2. BIFSG is
    implemented as follows:

    .. math::

        P(r|f,s,g) = \frac{P(r|s) \times P(f|r) \times P(g|r)}{\sum_{r=1}^4 P(r|s) \times P(f|r) \times P(g|r)}

    where `r` is race, `f` is first name, `s` is surname, and `g` is geography. The sum
    is across all races, i.e. Asian, Black, Hispanic, and White.

    Parameters
    ----------
    first_name: Name
        A string or array-like of strings
    last_name : Name
        A string or array-like of strings
    geography : Geography
        A scalar or array-like of geographies
    geo_type : GeoType
        One of `zcta` or `tract`

    Returns
    -------
    pd.DataFrame
        A DataFrame of last_name, geography, and `P(r|f,s,g)` for Asian, Black,
        Hispanic, and White. If either the first name, last name or geography cannot
        be found, the probability is `NaN`.

    Notes
    -----
    The data files can be found in:
        - data/distributionsprob_first_name_given_race.parquet
        - data/distributionsprob_race_given_last_name.parquet
        - data/distributionsprob_zcta_given_race_2010.parquet
        - data/distributionsprob_tract_given_race_2010.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.bifsg(
            first_name="cangyuan", last_name="li", zcta=27106, geo_type="zcta"
        )
    >>> pyethnicity.bifsg(
    >>> first_name=["cangyuan", "mark"],
    >>>     last_name=["li", "luo"],
    >>>     zcta=[27106, 11106],
    >>>     geo_type="zcta"
    >>> )
    """
    _assert_equal_lengths(first_name, last_name, geography)

    first_name_cleaned = _normalize_name(first_name, "first_name")
    last_name_cleaned = _normalize_name(last_name, "last_name")

    df = pl.concat([first_name_cleaned, last_name_cleaned], how="horizontal")

    prob_first_name_given_race: pl.DataFrame = df.join(
        RESOURCE_LOADER.load("prob_first_name_given_race"),
        left_on="first_name",
        right_on="name",
        how="left",
    ).select(RACES)

    prob_race_given_last_name: pl.DataFrame = df.join(
        RESOURCE_LOADER.load("prob_race_given_last_name"),
        left_on="last_name",
        right_on="name",
        how="left",
    ).select(RACES)

    prob_geo_given_race = _resolve_geography(geography, geo_type).select(RACES)

    bifsg_numer = (
        prob_first_name_given_race * prob_race_given_last_name * prob_geo_given_race
    )
    bifsg_denom = bifsg_numer.sum(axis=1)
    bifsg_probs = bifsg_numer / bifsg_denom

    df: pd.DataFrame = bifsg_probs.to_pandas()
    df.insert(0, "first_name", first_name)
    df.insert(1, "last_name", last_name)
    df.insert(2, geo_type, geography)

    return df
