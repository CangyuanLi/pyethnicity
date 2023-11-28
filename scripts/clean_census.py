from pathlib import Path

import polars as pl
import polars.selectors as cs

BASE_PATH = Path(__file__).resolve().parents[0]
DAT_PATH = BASE_PATH / "data"

# tidyr::pivot_wider(
#     id_cols = GEOID, names_from = variable, values_from = value
# ) %>%


NAME_MAPPER = {
    "P11_001N": "total",
    "P11_002N": "hispanic",
    "P11_003N": "nh_total",
    "P11_005N": "nh_white",
    "P11_006N": "nh_black",
    "P11_007N": "nh_aian",
    "P11_008N": "nh_asian",
    "P11_009N": "nh_hpi",
    "P11_010N": "nh_other",
    "P11_011N": "nh_multiracial",
    "P11_017N": "nh_white_other",
    "P11_021N": "nh_black_other",
    "P11_024N": "nh_aian_other",
    "P11_025N": "nh_asian_hpi",
    "P11_026N": "nh_asian_other",
    "P11_027N": "nh_hpi_other",
    "P11_048N": "nh_asian_hpi_other",
}


def main():
    df = (
        pl.read_parquet(DAT_PATH / "block_group/raw/*.parquet")
        .pivot(values="value", index="GEOID", columns="variable")
        .rename(NAME_MAPPER)
        .select(list(NAME_MAPPER.values()) + ["GEOID"])
        .with_columns(cs.numeric().cast(pl.Int64))
        .with_columns(state_fips=pl.col("GEOID").str.slice(0, 2))
        .rename({"GEOID": "block_group"})
    )
    df.write_parquet(DAT_PATH / "block_group/race_by_block_group_2020.parquet")

    df = (
        pl.read_parquet(DAT_PATH / "tract/raw/*.parquet")
        .pivot(values="value", index="GEOID", columns="variable")
        .rename(NAME_MAPPER)
        .select(list(NAME_MAPPER.values()) + ["GEOID"])
        .with_columns(cs.numeric().cast(pl.Int64))
        .with_columns(state_fips=pl.col("GEOID").str.slice(0, 2))
        .rename({"GEOID": "tract"})
    )
    df.write_parquet(DAT_PATH / "tract/race_by_tract_2020.parquet")

    df = (
        pl.read_parquet(DAT_PATH / "zcta/raw/zcta.parquet")
        .pivot(values="value", index="GEOID", columns="variable")
        .rename(NAME_MAPPER)
        .select(list(NAME_MAPPER.values()) + ["GEOID"])
        .with_columns(cs.numeric().cast(pl.Int64))
        .rename({"GEOID": "zcta"})
    )
    df.write_parquet(DAT_PATH / "zcta/race_by_zcta_2020.parquet")


if __name__ == "__main__":
    main()

    import pandas as pd

    pd.read_parquet(DAT_PATH / "zcta/race_by_zcta_2020.parquet").to_stata(
        BASE_PATH / "input_files/race_by_zcta_2020.dta", write_index=False
    )

    pd.read_parquet(DAT_PATH / "tract/race_by_tract_2020.parquet").to_stata(
        BASE_PATH / "input_files/race_by_tract_2020.dta", write_index=False
    )

    pd.read_parquet(DAT_PATH / "block_group/race_by_block_group_2020.parquet").to_stata(
        BASE_PATH / "input_files/race_by_block_group_2020.dta", write_index=False
    )
