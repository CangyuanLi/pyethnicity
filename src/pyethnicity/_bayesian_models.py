from __future__ import annotations

import string
import typing
from typing import Literal, Optional

import pandas as pd
import polars as pl

from .utils.paths import DIST_PATH
from .utils.types import Geography, GeoType, Name, Year
from .utils.utils import (
    RACES,
    RACES_6,
    _assert_equal_lengths,
    _download,
    _remove_single_chars,
)

UNWANTED_CHARS = string.digits + string.punctuation + string.whitespace

Resource = Literal[
    "prob_race_given_first_name",
    "prob_first_name_given_race",
    "prob_race_given_last_name",
    "prob_zcta_given_race_2010",
    "prob_race_given_zcta_2010",
    "prob_tract_given_race_2010",
    "prob_race_given_tract_2010",
    "ssa",
    "6cat/prob_race_given_first_name",
    "6cat/prob_first_name_given_race",
    "6cat/prob_race_given_last_name",
    "6cat/prob_race_given_zcta_2010",
    "6cat/prob_race_given_tract_2010",
]


class ResourceLoader:
    def __init__(self):
        self._resources: dict[Resource, Optional[pl.DataFrame]] = {
            k: None for k in typing.get_args(Resource)
        }

    def load(self, resource: Resource) -> pl.DataFrame:
        if self._resources[resource] is None:
            file = f"{resource}.parquet"
            if not (DIST_PATH / file).exists():
                _download(f"distributions/{file}")

            self._resources[resource] = pl.read_parquet(DIST_PATH / file)

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


def _bisg_internal(
    last_name: Name, geography: Geography, geo_type: GeoType, is_6cat: bool
) -> pd.DataFrame:
    if is_6cat:
        races = RACES_6
        prob_race_given_last_name_path = "6cat/prob_race_given_last_name"
    else:
        races = RACES
        prob_race_given_last_name_path = "prob_race_given_last_name"

    _assert_equal_lengths(last_name, geography)

    last_name_cleaned = _normalize_name(last_name, "last_name")

    prob_race_given_last_name = last_name_cleaned.join(
        RESOURCE_LOADER.load(prob_race_given_last_name_path),
        left_on="last_name",
        right_on="name",
        how="left",
    ).select(races)

    prob_geo_given_race = _resolve_geography(geography, geo_type).select(races)

    bisg_numer = prob_race_given_last_name * prob_geo_given_race
    bisg_denom = bisg_numer.sum(axis=1)
    bisg_probs = bisg_numer / bisg_denom

    df = bisg_probs.to_pandas()
    df.insert(0, "last_name", last_name)
    df.insert(1, geo_type, geography)

    return df


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
        - data/distributions/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2010.parquet
        - data/distributions/prob_tract_given_race_2010.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.bisg(last_name="li", zcta=27106, geo_type="zcta")
    >>> pyethnicity.bisg(last_name=["li", "luo"], zcta=[27106, 11106], geo_type="zcta")
    """
    return _bisg_internal(last_name, geography, geo_type, is_6cat=False)


def bisg6(last_name: Name, geography: Geography, geo_type: GeoType) -> pd.DataFrame:
    r"""Implements Bayesian Improved Surname Geocoding (BISG), developed by
    Elliot et. al (2009) https://link.springer.com/article/10.1007/s10742-009-0047-1.

    .. math::

        P(r|s,g) = \frac{P(r|s) \times P(g|r)}{\sum_{r=1}^6 P(r|s) \times P(g|r)}

    where `r` is race, `s` is surname, and `g` is geography. The sum is across all
    races, i.e. Asian, Black, Hispanic, Multiple, Native, and White.

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
        Hispanic, Multiple, Native and White. If either the last name or geography
        cannot be found, the probability is `NaN`.

    Notes
    -----
    The data files can be found in:
        - data/distributions/6cat/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2010.parquet
        - data/distributions/prob_tract_given_race_2010.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.bisg6(last_name="li", zcta=27106, geo_type="zcta")
    >>> pyethnicity.bisg6(last_name=["li", "luo"], zcta=[27106, 11106], geo_type="zcta")
    """
    return _bisg_internal(last_name, geography, geo_type, is_6cat=True)


def _bifsg_internal(
    first_name: Name,
    last_name: Name,
    geography: Geography,
    geo_type: GeoType,
    is_6cat: bool,
) -> pd.DataFrame:
    if is_6cat:
        races = RACES_6
        prefix = "6cat/"
    else:
        races = RACES
        prefix = ""

    prob_first_name_given_race_path = f"{prefix}prob_first_name_given_race"
    prob_race_given_last_name_path = f"{prefix}prob_race_given_last_name"

    _assert_equal_lengths(first_name, last_name, geography)

    first_name_cleaned = _normalize_name(first_name, "first_name")
    last_name_cleaned = _normalize_name(last_name, "last_name")

    df = pl.concat([first_name_cleaned, last_name_cleaned], how="horizontal")

    prob_first_name_given_race: pl.DataFrame = df.join(
        RESOURCE_LOADER.load(prob_first_name_given_race_path),
        left_on="first_name",
        right_on="name",
        how="left",
    ).select(races)

    prob_race_given_last_name: pl.DataFrame = df.join(
        RESOURCE_LOADER.load(prob_race_given_last_name_path),
        left_on="last_name",
        right_on="name",
        how="left",
    ).select(races)

    prob_geo_given_race = _resolve_geography(geography, geo_type).select(races)

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
        - data/distributions/prob_first_name_given_race.parquet
        - data/distributions/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2010.parquet
        - data/distributions/prob_tract_given_race_2010.parquet

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
    return _bifsg_internal(first_name, last_name, geography, geo_type, is_6cat=False)


def bifsg6(
    first_name: Name, last_name: Name, geography: Geography, geo_type: GeoType
) -> pd.DataFrame:
    r"""Implements Bayesian Improved Firstname Surname Geocoding (BIFSG), developed by
    Voicu (2018) https://www.tandfonline.com/doi/full/10.1080/2330443X.2018.1427012.
    Note that when using 6 categories, only the Voicu data is used. BIFSG is implemented
    as follows:

    .. math:

        P(r|f,s,g) = \frac{P(r|s) \times P(f|r) \times P(g|r)}{\sum_{r=1}^6 P(r|s) \times P(f|r) \times P(g|r)}

    where `r` is race, `f` is first name, `s` is surname, and `g` is geography. The sum
    is across all races, i.e. Asian, Black, Hispanic, Multiple, Native, and White.

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
        Hispanic, Multiple, Native, and White. If either the first name, last name or
        geography cannot be found, the probability is `NaN`.

    Notes
    -----
    The data files can be found in:
        - data/distributions/6cat/prob_first_name_given_race.parquet
        - data/distributions/6cat/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2010.parquet
        - data/distributions/prob_tract_given_race_2010.parquet

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
    return _bifsg_internal(first_name, last_name, geography, geo_type, is_6cat=True)


def _calc_correx(female: int, male: int) -> tuple[float, float]:
    ratio_female = female / (male + female)
    ratio_male = 1 - ratio_female

    return (0.5 / ratio_female, 0.5 / ratio_male)


def _get_correction_factor(
    df: pl.DataFrame, min_years: list[int], max_years: list[int]
) -> pl.DataFrame:
    res_min_year = []
    res_max_year = []
    female_correx = []
    male_correx = []

    year_df = (
        pl.LazyFrame({"min_year": min_years, "max_year": max_years}).unique().collect()
    )
    for min_year, max_year in year_df.iter_rows():
        subset = df.filter(pl.col("year").is_between(min_year, max_year, closed="both"))
        female = subset.get_column("count_female").sum()
        male = subset.get_column("count_male").sum()

        try:
            correx = _calc_correx(female, male)
        except ZeroDivisionError:
            correx = (1, 1)
        fc, mc = correx

        res_min_year.append(min_year)
        res_max_year.append(max_year)
        female_correx.append(fc)
        male_correx.append(mc)

    return pl.DataFrame(
        {
            "min_year": res_min_year,
            "max_year": res_max_year,
            "female_correx": female_correx,
            "male_correx": male_correx,
        }
    )


def predict_sex_ssa(
    first_name: Name,
    min_year: Year = 1880,
    max_year: Year = 2022,
    correct_skew: bool = True,
) -> pd.DataFrame:
    """Predicts sex from first name and a year range using Social Security
    Administration data. It simply calculates the proportion of people with a certain
    name in a certain year range that are male and female.

    Parameters
    ----------
    first_name : Name
        A string or array-like of strings
    min_year : Year, optional
        An int or array-like of ints, by default 1880
    max_year : Year, optional
        An int or array-like of ints, by default 2022
    correct_skew : bool, optional
        Whether to correct the skew in the SSA data, by default True

    Returns
    -------
    pd.DataFrame
        A DataFrame of first_name, min_year, max_year, pct_female, and pct_male.

    Notes
    -----
    The data files can be found in:
        - data/distributions/ssa.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.predict_sex_ssa(first_name="john")
    >>> pyethnicity.predict_sex_ssa(first_name=["john", "mary"], min_year=[1880, 1990])
    """
    if isinstance(first_name, str):
        first_name = [first_name]

    # broadcast years
    target_len = len(first_name)

    if isinstance(min_year, int):
        min_year = [min_year for _ in range(target_len)]

    if isinstance(max_year, int):
        max_year = [max_year for _ in range(target_len)]

    _assert_equal_lengths(first_name, min_year, max_year)

    # create dataframe of inputs to merge on
    inputs = (
        pl.LazyFrame(
            {"first_name": first_name, "min_year": min_year, "max_year": max_year}
        )
        .with_columns(first_name_clean=pl.col("first_name").str.to_lowercase())
        .unique(["first_name", "min_year", "max_year"])
    )

    ssa = RESOURCE_LOADER.load("ssa")

    df = (
        inputs.join(
            ssa.lazy(), left_on="first_name_clean", right_on="first_name", how="left"
        )
        .filter(
            pl.col("year").is_between(
                pl.col("min_year"), pl.col("max_year"), closed="both"
            )
            | pl.col("year").is_null()
        )
        .groupby("first_name", "min_year", "max_year")
        .agg(pl.col("count_female", "count_male").sum())
    )

    if correct_skew:
        correx = _get_correction_factor(ssa, min_year, max_year)
        df = df.join(correx.lazy(), on=["min_year", "max_year"], how="left")
    else:
        df.with_columns(female_correx=pl.lit(1), male_correx=pl.lit(1))

    res = (
        df.with_columns(
            pl.col("count_female") * pl.col("female_correx"),
            pl.col("count_male") * pl.col("male_correx"),
        )
        .with_columns(total=pl.col("count_female") + pl.col("count_male"))
        .with_columns(
            pct_female=pl.col("count_female") / pl.col("total"),
            pct_male=pl.col("count_male") / pl.col("total"),
        )
        .select("first_name", "min_year", "max_year", "pct_female", "pct_male")
    )

    return res.collect().to_pandas()
