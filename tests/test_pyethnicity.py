import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import surgeo

import pyethnicity

BASE_PATH = Path(__file__).resolve().parents[0]

ZCTA = 11106
TRACT = 72153750502
RACES = ["asian", "black", "hispanic", "white"]

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


# def test_bisg6():
#     key = ["last_name", "zcta"]
#     surgeo_preds = (
#         surgeo.SurgeoModel()
#         .get_probabilities(PPP["last_name"], PPP["zcta"])
#         .rename(columns={"name": "last_name", "zcta5": "zcta"})
#     )
#     pyeth_preds = (
#         pyethnicity.bisg(PPP["last_name"], PPP["zcta"], geo_type="zcta")
#         .drop_duplicates(key)
#         .sort_values(key)
#     )

#     res = pd.merge(pyeth_preds, surgeo_preds, on=["last_name", "zcta"], how="inner")
#     res["diff"] = res["black_x"] - res["black_y"]
#     res[["last_name", "zcta", "black_x", "black_y", "diff"]].to_csv("temp.csv")

#     assert np.isclose(
#         res["black_x"].to_numpy(), res["black_y"].to_numpy(), equal_nan=True
#     ).all()


# def test_bifsg6():
#     pass


def test_bisg():
    df = pyethnicity.bisg("luo", TRACT, "tract")
    assert arr_equal(df.columns, ["last_name", "tract"] + RACES)

    df = pyethnicity.bisg("luo", ZCTA, "zcta")
    assert arr_equal(df.columns, ["last_name", "zcta"] + RACES)

    df = df.drop("last_name", axis=1)
    for other in ["Luo", "lUo", "luo jr."]:
        assert df.equals(
            pyethnicity.bisg(other, ZCTA, "zcta").drop("last_name", axis=1)
        )


def test_bifsg():
    df = pyethnicity.bifsg("mercy", "luo", TRACT, "tract")
    assert arr_equal(df.columns, ["first_name", "last_name", "tract"] + RACES)

    df = pyethnicity.bifsg("mercy", "luo", ZCTA, "zcta")
    assert arr_equal(df.columns, ["first_name", "last_name", "zcta"] + RACES)

    df = df.drop(["first_name", "last_name"], axis=1)
    for other_fn, other_ln in zip(
        ["MErcY", "mercy12", "mercy sr."], ["Luo", "lUo", "luo jr."]
    ):
        assert df.equals(
            pyethnicity.bifsg(other_fn, other_ln, ZCTA, "zcta").drop(
                ["first_name", "last_name"], axis=1
            )
        )


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
