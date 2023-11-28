from pathlib import Path

import pandas as pd
import polars as pl
import polars.selectors as cs

CREATED_PATH = Path(__file__).parent / "created_files"
DIST_PATH = Path(__file__).resolve().parents[1] / "src/pyethnicity/data/distributions"


def main():
    for f in CREATED_PATH.glob("*.dta"):
        filename = f.stem
        geo = filename.split("_")[2]

        if geo == "block":
            geo = "block_group"

        df = pl.from_pandas(pd.read_stata(f))

        if geo == "zcta":
            df = df.rename({"zcta": "zcta5"})
            geo_col = "zcta5"
        else:
            geo_col = geo

        name_mapper = {"mult_other": "multiple", "api": "asian", "aian": "native"}

        prob_race_given_geo = df.select(
            geo_col, cs.starts_with("geo_").name.map(lambda c: c.replace("geo_pr_", ""))
        ).rename(name_mapper)
        prob_geo_given_race = df.select(
            geo_col,
            cs.starts_with("here_").name.map(lambda c: c.replace("here_given_", "")),
        ).rename(name_mapper)

        prob_race_given_geo.write_csv(CREATED_PATH / f"prob_race_given_{geo}_2020.csv")
        prob_geo_given_race.write_csv(CREATED_PATH / f"prob_{geo}_given_race_2020.csv")

        prob_race_given_geo.write_parquet(
            DIST_PATH / f"prob_race_given_{geo}_2020.parquet"
        )
        prob_geo_given_race.write_parquet(
            DIST_PATH / f"prob_{geo}_given_race_2020.parquet"
        )


if __name__ == "__main__":
    main()
