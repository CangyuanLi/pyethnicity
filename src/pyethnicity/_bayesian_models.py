from __future__ import annotations

import functools
import string
import typing
from collections.abc import Iterable
from typing import Literal, Optional

import polars as pl
import polars.selectors as cs

from .utils.paths import DIST_PATH
from .utils.types import Geography, GeoType, Name, Year
from .utils.utils import (
    RACES,
    RACES_6,
    IntoExpr,
    _as_expr,
    _download,
    _remove_single_chars,
    _set_name,
)

UNWANTED_CHARS = string.digits + string.punctuation + string.whitespace

Resource = Literal[
    "prob_race_given_first_name",
    "prob_first_name_given_race",
    "prob_race_given_last_name",
    "prob_zcta_given_race_2020",
    "prob_race_given_zcta_2020",
    "prob_tract_given_race_2020",
    "prob_race_given_tract_2020",
    "prob_block_group_given_race_2020",
    "prob_race_given_block_group_2020",
    "ssa",
    "6cat/prob_race_given_first_name",
    "6cat/prob_first_name_given_race",
    "6cat/prob_race_given_last_name",
]


class ResourceLoader:
    def __init__(self):
        self._resources: dict[Resource, Optional[pl.LazyFrame]] = {
            k: None for k in typing.get_args(Resource)
        }

    def load(self, resource: Resource) -> pl.LazyFrame:
        if self._resources[resource] is None:
            file = f"{resource}.parquet"
            if not (DIST_PATH / file).exists():
                _download(f"distributions/{file}")

            self._resources[resource] = pl.scan_parquet(DIST_PATH / file)

        data = self._resources[resource]
        assert data is not None

        return data


RESOURCE_LOADER = ResourceLoader()

WaterfallJoinType = Literal["left", "inner"]


def _waterfall_join(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    left_on: Iterable[str],
    right_on: str,
    how: WaterfallJoinType = "left",
) -> pl.LazyFrame:
    left = left.with_row_count("index")
    seen: list[int] = []
    outputs: list[pl.LazyFrame] = []

    for col in left_on:
        output = left.filter(~pl.col("index").is_in(seen)).join(
            right, left_on=col, right_on=right_on, how="inner"
        )

        seen.extend(
            output.select("index").lazy().collect().get_column("index").to_list()
        )
        outputs.append(output)

    if how == "left":
        outputs.append(left.filter(~pl.col("index").is_in(seen)))

    return pl.concat(outputs, how="diagonal").sort("index").drop("index")


def _waterfall_fill(*exprs: IntoExpr) -> pl.Expr:
    exprs = [_as_expr(expr) for expr in exprs]

    def func(acc: pl.Expr, x: pl.Expr):
        return pl.when(acc.is_null()).then(x).otherwise(acc)

    return functools.reduce(func, exprs)


def _sort_geo_cols(cols: tuple[str]) -> list[str]:
    ranks = []
    for col in cols:
        if "block_group" in col:
            ranks.append(0)
        elif "tract" in col:
            ranks.append(1)
        elif "zcta" in col:
            ranks.append(2)

    return [x for _, x in sorted(zip(ranks, cols))]


def _remove_chars(expr: pl.Expr) -> pl.Expr:
    for char in UNWANTED_CHARS:
        expr = expr.str.replace_all(char, "", literal=True)

    return expr


def _normalize_name(expr: pl.Expr) -> pl.Expr:
    return (
        expr.str.to_uppercase()
        .str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?III\s*?$", "")
        .str.replace_all(r"\s?IV\s*?$", "")
        .pipe(_remove_chars)
        .pipe(_remove_single_chars)
    )


def _split_name(lf: pl.LazyFrame, col_name: str) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col(col_name)
        .str.split_exact("-", 1)
        .struct.rename_fields([f"{col_name}_1", f"{col_name}_2"])
        .alias("fields")
    ).unnest("fields")


def _normalize_zcta(zcta: Geography, col_name: str = "zcta5") -> pl.LazyFrame:
    return pl.LazyFrame({col_name: zcta}).with_columns(
        pl.col(col_name).cast(str).str.zfill(5)
    )


def _normalize_tract(tract: Geography, col_name: str = "tract") -> pl.LazyFrame:
    return pl.LazyFrame({col_name: tract}).with_columns(
        pl.col(col_name).cast(str).str.zfill(11)
    )


def _normalize_block_group(
    block_group: Geography, col_name: str = "block_group"
) -> pl.LazyFrame:
    return pl.LazyFrame({col_name: block_group}).with_columns(
        pl.col(col_name).cast(str).str.zfill(12)
    )


# TODO: It seems a little wasteful to extract the geography this way now that we have
# moved to putting everything in a LazyFrame upfront.
def _resolve_geography(geography: Geography, geo_type: GeoType) -> pl.LazyFrame:
    if geo_type == "tract":
        geo = _normalize_tract(geography)
        prob_geo_given_race = geo.join(
            RESOURCE_LOADER.load("prob_tract_given_race_2020"), on="tract", how="left"
        )
    elif geo_type == "zcta":
        geo = _normalize_zcta(geography)
        prob_geo_given_race = geo.join(
            RESOURCE_LOADER.load("prob_zcta_given_race_2020"), on="zcta5", how="left"
        )
    elif geo_type == "block_group":
        geo = _normalize_block_group(geography)
        prob_geo_given_race = geo.join(
            RESOURCE_LOADER.load("prob_block_group_given_race_2020"),
            on="block_group",
            how="left",
        )
    else:
        raise ValueError(f"`{geo_type}` is not a valid geography.")

    return prob_geo_given_race


def _bng(
    prob_race_given_name: pl.DataFrame,
    zcta: Geography,
    tract: Geography,
    block_group: Geography,
) -> pl.DataFrame:
    name_mapper = {
        "zcta": _set_name(zcta, "zcta"),
        "tract": _set_name(tract, "tract"),
        "block_group": _set_name(block_group, "block_group"),
    }
    data = {"zcta": zcta, "tract": tract, "block_group": block_group}
    data = {k: v for k, v in data.items() if v is not None}

    if not data:
        raise ValueError("At least one geography must be specified.")

    valid_geo_types = list(data.keys())

    prob_list = []
    for geo_type in valid_geo_types:
        prob_geo_given_race = (
            _resolve_geography(data[geo_type], geo_type).select(RACES).collect()
        )

        numer = prob_race_given_name.select(RACES) * prob_geo_given_race.select(RACES)
        denom = numer.sum_horizontal()
        probs = numer / denom
        prob_list.append(
            probs.select(pl.col(RACES).name.map(lambda c: f"{geo_type}_{c}")).fill_nan(
                None
            )
        )

    df: pl.DataFrame = pl.concat(prob_list, how="horizontal")

    for race in RACES:
        cols = _sort_geo_cols(cs.expand_selector(df, cs.ends_with(race)))
        df = df.with_columns(_waterfall_fill(*cols).alias(race))

    df = df.drop(cs.contains(valid_geo_types))

    return pl.concat(
        [
            prob_race_given_name.drop(RACES),
            pl.DataFrame(data).rename(name_mapper),
            probs,
        ],
        how="horizontal",
    )


def _bisg_internal(
    last_name: Name,
    zcta: Optional[Geography],
    tract: Optional[Geography],
    block_group: Optional[Geography],
    drop_intermediate: bool,
    is_6cat: bool,
) -> pl.DataFrame:
    if is_6cat:
        races = RACES_6
        prob_race_given_last_name_path = "6cat/prob_race_given_last_name"
    else:
        races = RACES
        prob_race_given_last_name_path = "prob_race_given_last_name"

    name_mapper = {
        "last_name": _set_name(last_name, "last_name"),
        "zcta": _set_name(zcta, "zcta"),
        "tract": _set_name(tract, "tract"),
        "block_group": _set_name(block_group, "block_group"),
    }

    data = {"zcta": zcta, "tract": tract, "block_group": block_group}
    data = {k: v for k, v in data.items() if v is not None}

    if not data:
        raise ValueError("At least one geography must be specified.")

    valid_geo_types = list(data.keys())

    data["last_name_raw"] = last_name

    raw = (
        pl.LazyFrame(data)
        .drop_nulls(subset=["last_name_raw"])
        .filter(~pl.all_horizontal(pl.col(valid_geo_types).is_null()))
        .unique()
        .with_row_index("index")
        .pipe(_split_name, "last_name_raw")
        .with_columns(
            pl.col("last_name_raw", "last_name_raw_1", "last_name_raw_2")
            .pipe(_normalize_name)
            .name.map(lambda x: x.replace("_raw", "_clean"))
        )
        .collect()
    )

    clean_last_name_cols = ["last_name_clean", "last_name_clean_1", "last_name_clean_2"]
    last_name_cleaned = raw.select(clean_last_name_cols)

    prob_race_given_last_name = (
        _waterfall_join(
            last_name_cleaned.lazy(),
            RESOURCE_LOADER.load(prob_race_given_last_name_path),
            left_on=clean_last_name_cols,
            right_on="name",
            how="left",
        )
        .select(races)
        .collect()
    )

    probs = []
    for geo_type in valid_geo_types:
        prob_geo_given_race = (
            _resolve_geography(raw[geo_type], geo_type).select(races).collect()
        )

        bisg_numer = prob_race_given_last_name * prob_geo_given_race
        bisg_denom = bisg_numer.sum_horizontal()
        bisg_probs = bisg_numer / bisg_denom

        probs.append(
            bisg_probs.select(
                pl.col(races).name.map(lambda c: f"{geo_type}_{c}")
            ).fill_nan(None)
        )

    df: pl.DataFrame = pl.concat(probs, how="horizontal")

    for race in races:
        cols = _sort_geo_cols(cs.expand_selector(df, cs.ends_with(race)))
        df = df.with_columns(_waterfall_fill(*cols).alias(race))

    if drop_intermediate:
        df = df.drop(cs.contains(valid_geo_types))

    # final bookeeping
    df.insert_column(0, raw["last_name_raw"].rename(name_mapper["last_name"]))

    for idx, geo_type in enumerate(valid_geo_types, start=1):
        df.insert_column(idx, raw[geo_type].rename(name_mapper[geo_type]))

    return df


def bisg(
    last_name: Name,
    zcta: Optional[Geography] = None,
    tract: Optional[Geography] = None,
    block_group: Optional[Geography] = None,
    drop_intermediate: bool = True,
) -> pl.DataFrame:
    r"""Implements Bayesian Improved Surname Geocoding (BISG), developed by
    Elliot et. al (2009) https://link.springer.com/article/10.1007/s10742-009-0047-1.
    Pyethnicity augments the Census surname list with distributions calculated from
    voter registration data sourced from L2.

    .. math::

        P(r|s,g) = \frac{P(r|s) \times P(g|r)}{\sum_{r=1}^4 P(r|s) \times P(g|r)}

    where `r` is race, `s` is surname, and `g` is geography. The sum is across all
    races, i.e. Asian, Black, Hispanic, and White. You may specify
    multiple geographies. If so, the resulting predictions are waterfalled together,
    where the priority is block group -> tract -> zcta.

    Parameters
    ----------
    last_name : Name
        A string or array-like of strings
    zcta : Geography, optional
        A scalar or array-like of Census ZCTAs, by default None
    tract : Geography, optional
        A scalar or array-like of Census Tracts, by default None
    block_group : Geography, optional
        A scalar or array-like of Census Block Groups, by default None
    drop_intermediate : bool
        Whether to drop intermediate calculations, by default True

    Returns
    -------
    pd.DataFrame
        A DataFrame of last name, geography, and `P(r | s, g)` for Asian, Black,
        Hispanic, and White. If either the last name or geography cannot be found,
        the probability is null.

    Notes
    -----
    The data files can be found in:
        - data/distributions/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2020.parquet
        - data/distributions/prob_tract_given_race_2020.parquet
        - data/distributions/prob_block_group_given_race_2020.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.bisg(last_name="li", zcta=27106, geo_type="zcta")
    >>> pyethnicity.bisg(last_name=["li", "luo"], zcta=[27106, 11106], geo_type="zcta")
    """
    return _bisg_internal(
        last_name, zcta, tract, block_group, drop_intermediate, is_6cat=False
    )


def bisg6(
    last_name: Name,
    zcta: Optional[Geography] = None,
    tract: Optional[Geography] = None,
    block_group: Optional[Geography] = None,
    drop_intermediate: bool = True,
) -> pl.DataFrame:
    r"""Implements Bayesian Improved Surname Geocoding (BISG), developed by
    Elliot et. al (2009) https://link.springer.com/article/10.1007/s10742-009-0047-1.

    .. math::

        P(r|s,g) = \frac{P(r|s) \times P(g|r)}{\sum_{r=1}^6 P(r|s) \times P(g|r)}

    where `r` is race, `s` is surname, and `g` is geography. The sum is across all
    races, i.e. Asian, Black, Hispanic, Multiple, Native, and White. You may specify
    multiple geographies. If so, the resulting predictions are waterfalled together,
    where the priority is block group -> tract -> zcta.

    Parameters
    ----------
    last_name : Name
        A string or array-like of strings
    zcta : Geography, optional
        A scalar or array-like of Census ZCTAs, by default None
    tract : Geography, optional
        A scalar or array-like of Census Tracts, by default None
    block_group : Geography, optional
        A scalar or array-like of Census Block Groups, by default None
    drop_intermediate : bool
        Whether to drop intermediate calculations, by default True

    Returns
    -------
    pd.DataFrame
        A DataFrame of last name, geographies, and `P(r | s, g)` for Asian, Black,
        Hispanic, Multiple, Native and White. If either the last name or all geographies
        cannot be found, the probability is null.

    Notes
    -----
    The data files can be found in:
        - data/distributions/6cat/prob_race_given_last_name.parquet
        - data/distributions/prob_zcta_given_race_2020.parquet
        - data/distributions/prob_tract_given_race_2020.parquet
        - data/distributions/prob_block_group_given_race_2020.parquet

    Examples
    --------
    >>> import pyethnicity
    >>> pyethnicity.bisg6(last_name="li", zcta=27106, geo_type="zcta")
    >>> pyethnicity.bisg6(last_name=["li", "luo"], zcta=[27106, 11106], geo_type="zcta")
    """
    return _bisg_internal(
        last_name, zcta, tract, block_group, drop_intermediate, is_6cat=True
    )


def _bifsg_internal(
    first_name: Name,
    last_name: Name,
    zcta: Optional[Geography],
    tract: Optional[Geography],
    block_group: Optional[Geography],
    drop_intermediate: bool,
    is_6cat: bool,
) -> pl.DataFrame:
    if is_6cat:
        races = RACES_6
        prefix = "6cat/"
    else:
        races = RACES
        prefix = ""

    prob_first_name_given_race_path = f"{prefix}prob_first_name_given_race"
    prob_race_given_last_name_path = f"{prefix}prob_race_given_last_name"

    name_mapper = {
        "first_name": _set_name(first_name, "first_name"),
        "last_name": _set_name(last_name, "last_name"),
        "zcta": _set_name(zcta, "zcta"),
        "tract": _set_name(tract, "tract"),
        "block_group": _set_name(block_group, "block_group"),
    }

    data = {"zcta": zcta, "tract": tract, "block_group": block_group}
    data = {k: v for k, v in data.items() if v is not None}

    if not data:
        raise ValueError("At least one geography must be specified.")

    valid_geo_types = list(data.keys())

    data["first_name_raw"] = first_name
    data["last_name_raw"] = last_name

    raw = (
        pl.LazyFrame(data)
        .drop_nulls(subset=["last_name_raw"])
        .filter(~pl.all_horizontal(pl.col(valid_geo_types).is_null()))
        .unique()
        .with_row_index("index")
        .pipe(_split_name, "last_name_raw")
        .with_columns(
            pl.col(
                "last_name_raw", "last_name_raw_1", "last_name_raw_2", "first_name_raw"
            )
            .pipe(_normalize_name)
            .name.map(lambda x: x.replace("_raw", "_clean")),
        )
        .collect()
    )

    first_name_cleaned = raw.select("first_name_clean")

    clean_last_name_cols = ["last_name_clean", "last_name_clean_1", "last_name_clean_2"]
    last_name_cleaned = raw.select(clean_last_name_cols)

    prob_race_given_last_name = (
        _waterfall_join(
            last_name_cleaned.lazy(),
            RESOURCE_LOADER.load(prob_race_given_last_name_path),
            left_on=clean_last_name_cols,
            right_on="name",
            how="left",
        )
        .select(races)
        .collect()
    )

    prob_first_name_given_race: pl.DataFrame = first_name_cleaned.join(
        RESOURCE_LOADER.load(prob_first_name_given_race_path).collect(),
        left_on="first_name_clean",
        right_on="name",
        how="left",
    ).select(races)

    probs = []
    for geo_type in valid_geo_types:
        prob_geo_given_race = (
            _resolve_geography(raw[geo_type], geo_type).select(races).collect()
        )

        bifsg_numer = (
            prob_first_name_given_race * prob_race_given_last_name * prob_geo_given_race
        )
        bifsg_denom = bifsg_numer.sum_horizontal()
        bifsg_probs = bifsg_numer / bifsg_denom

        probs.append(
            bifsg_probs.select(
                pl.col(races).name.map(lambda c: f"{geo_type}_{c}")
            ).fill_nan(None)
        )

    df: pl.DataFrame = pl.concat(probs, how="horizontal")

    for race in races:
        cols = _sort_geo_cols(cs.expand_selector(df, cs.ends_with(race)))
        df = df.with_columns(_waterfall_fill(*cols).alias(race))

    if drop_intermediate:
        df = df.drop(cs.contains(valid_geo_types))

    # final bookeeping
    df.insert_column(0, raw["first_name_raw"].rename(name_mapper["first_name"]))
    df.insert_column(1, raw["last_name_raw"].rename(name_mapper["last_name"]))

    for idx, geo_type in enumerate(valid_geo_types, start=2):
        df.insert_column(idx, raw[geo_type].rename(name_mapper[geo_type]))

        prob_geo_given_race = (
            _resolve_geography(raw[geo_type], geo_type).select(races).collect()
        )

    return df


def bifsg(
    first_name: Name,
    last_name: Name,
    zcta: Optional[Geography] = None,
    tract: Optional[Geography] = None,
    block_group: Optional[Geography] = None,
    drop_intermediate: bool = True,
) -> pl.DataFrame:
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
        - data/distributions/prob_zcta_given_race_2020.parquet
        - data/distributions/prob_tract_given_race_2020.parquet
        - data/distributions/prob_block_group_given_race_2020.parquet

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
    return _bifsg_internal(
        first_name,
        last_name,
        zcta,
        tract,
        block_group,
        drop_intermediate,
        is_6cat=False,
    )


def bifsg6(
    first_name: Name,
    last_name: Name,
    zcta: Optional[Geography] = None,
    tract: Optional[Geography] = None,
    block_group: Optional[Geography] = None,
    drop_intermediate: bool = True,
) -> pl.DataFrame:
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
        - data/distributions/prob_zcta_given_race_2020.parquet
        - data/distributions/prob_tract_given_race_2020.parquet
        - data/distributions/prob_block_group_given_race_2020.parquet

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
    return _bifsg_internal(
        first_name,
        last_name,
        zcta,
        tract,
        block_group,
        drop_intermediate,
        is_6cat=True,
    )


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
) -> pl.DataFrame:
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
    name_mapper = {
        "first_name": _set_name(first_name, "first_name"),
        "min_year": _set_name(min_year, "min_year"),
        "max_year": _set_name(max_year, "max_year"),
    }

    if name_mapper["min_year"] == name_mapper["max_year"]:
        name_mapper["min_year"] = "min_year"
        name_mapper["max_year"] = "max_year"

    # create dataframe of inputs to merge on
    inputs = (
        pl.LazyFrame(
            {"first_name": first_name, "min_year": min_year, "max_year": max_year}
        )
        .with_columns(first_name_clean=pl.col("first_name").str.to_lowercase())
        .unique()
    )

    ssa = RESOURCE_LOADER.load("ssa")

    df = (
        inputs.join(ssa, left_on="first_name_clean", right_on="first_name", how="left")
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
        correx = _get_correction_factor(ssa.collect(), min_year, max_year)
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
        .rename(name_mapper)
    )

    return res.collect()
