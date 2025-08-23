import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import surgeo

import pyethnicity
import pyethnicity.utils.utils as pyeth_utils

BASE_PATH = Path(__file__).resolve().parents[0]

# ZCTA = 11106
# TRACT = 72153750502
# RACES = ["asian", "black", "hispanic", "white"]

PPP = (
    pl.scan_parquet(BASE_PATH / "ppp_test.parquet")
    .select("first_name", "last_name", "zcta")
    .with_columns(pl.col("first_name", "last_name").str.to_uppercase())
    .collect()
    .sample(10_000)
    .to_pandas()
)


def arr_equal(arr1, arr2):
    for i, j in zip(arr1, arr2):
        if i != j:
            return False

    return True


def test_set_name():
    assert pyeth_utils._set_name(["mercy"], "first_name") == "first_name"
    assert pyeth_utils._set_name(pl.Series("fname", ["mercy"]), "first_name") == "fname"
    assert (
        pyeth_utils._set_name(
            PPP.rename(columns={"first_name": "blah"})["blah"], "first_name"
        )
        == "blah"
    )


def test_remove_single_chars():
    df = pl.DataFrame(
        {
            "name": ["james a denise", "james denise", "a james a a c denise d"],
            "name_expected": "james denise",
        }
    ).with_columns(pyeth_utils._remove_single_chars("name").alias("name_cleaned"))

    assert (df["name_cleaned"] == df["name_expected"]).all()


def test_sort_geo_cols():
    from pyethnicity._bayesian_models import _sort_geo_cols

    assert _sort_geo_cols(["tract"]) == ["tract"]
    assert _sort_geo_cols(["tract", "block_group", "zcta"]) == [
        "block_group",
        "tract",
        "zcta",
    ]
    assert _sort_geo_cols(["zcta_asian", "block_group_asian"]) == [
        "block_group_asian",
        "zcta_asian",
    ]


def test_assert_equal_lengths():
    assert pyeth_utils._assert_equal_lengths("mercy", "luo", 27106) is None

    with pytest.raises(ValueError):
        pyeth_utils._assert_equal_lengths("mercy", ["luo", "li"])

    assert (
        pyeth_utils._assert_equal_lengths(
            ["luo", "li"], pd.Series(["luo", "li"], pl.Series("x", ["luo", "li"]))
        )
        is None
    )


def test_bisg6():
    key = ["last_name", "zcta"]
    # surgeo_preds = (
    #     surgeo.SurgeoModel()
    #     .get_probabilities(PPP["last_name"], PPP["zcta"])
    #     .rename(columns={"name": "last_name", "zcta5": "zcta"})
    # )
    pyeth_preds = pyethnicity.bisg6(PPP["last_name"], zcta=PPP["zcta"])

    # res = pd.merge(pyeth_preds, surgeo_preds, on=["last_name", "zcta"], how="inner")
    # res["diff"] = res["black_x"] - res["black_y"]
    # res[["last_name", "zcta", "black_x", "black_y", "diff"]].to_csv("temp.csv")

    # assert np.isclose(
    #     res["black_x"].to_numpy(), res["black_y"].to_numpy(), equal_nan=True
    # ).all()


def test_bifsg6():
    pyethnicity.bifsg6(PPP["first_name"], PPP["last_name"], zcta=PPP["zcta"])


def test_bisg():
    pyethnicity.bisg(PPP["last_name"], zcta=PPP["zcta"])


def test_bifsg():
    pyethnicity.bifsg(PPP["first_name"], PPP["last_name"], zcta=PPP["zcta"])


def test_readme():
    zcta = 27106

    tract = 72153750502
    first_name = "cangyuan"
    last_name = "luo"

    pyethnicity.bisg(last_name, zcta=zcta)
    pyethnicity.bifsg(first_name, last_name, zcta=zcta, tract=tract)
    pyethnicity.predict_race_fl(first_name, last_name)
    pyethnicity.predict_race_flg(first_name, last_name, tract=tract)
    pyethnicity.predict_race(first_name, last_name, zcta=zcta)


# def test_bifsg():
#     df = pyethnicity.bifsg("mercy", "luo", TRACT, "tract")
#     assert arr_equal(df.columns, ["first_name", "last_name", "tract"] + RACES)

#     df = pyethnicity.bifsg("mercy", "luo", ZCTA, "zcta")
#     assert arr_equal(df.columns, ["first_name", "last_name", "zcta"] + RACES)

#     df = df.drop(["first_name", "last_name"], axis=1)
#     for other_fn, other_ln in zip(
#         ["MErcY", "mercy12", "mercy sr."], ["Luo", "lUo", "luo jr."]
#     ):
#         assert df.equals(
#             pyethnicity.bifsg(other_fn, other_ln, ZCTA, "zcta").drop(
#                 ["first_name", "last_name"], axis=1
#             )
#         )


# def test_ssa():
#     r_results = (
#         pl.scan_csv(BASE_PATH / "gender_r_package_results.csv")
#         .rename(
#             {
#                 "name": "first_name",
#                 "year_min": "min_year",
#                 "year_max": "max_year",
#                 "proportion_male": "pct_male_r",
#                 "proportion_female": "pct_female_r",
#             }
#         )
#         .drop("gender")
#         .collect()
#     )

#     py_results = pyethnicity.predict_sex_ssa(
#         r_results.get_column("first_name"), min_year=1990, max_year=2000
#     )

#     df = r_results.join(
#         pl.from_pandas(py_results),
#         on=["first_name", "min_year", "max_year"],
#         how="left",
#     )
#     print(df)

#     for pct_female_r, pct_male_r, pct_female, pct_male in df.select(
#         "pct_female_r", "pct_male_r", "pct_female", "pct_male"
#     ).iter_rows():
#         assert math.isclose(pct_female_r, pct_female)
#         assert math.isclose(pct_male_r, pct_male)
