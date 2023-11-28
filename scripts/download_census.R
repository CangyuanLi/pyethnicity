library(arrow)
library(dplyr)
library(stringr)
library(tidycensus)
library(tidyr)

CENSUS_PATH <- file.path("/home/cangyuanli/Documents/Projects/pyethnicity/scripts/data")

tidycensus::census_api_key("4067c86fb24937a0063e3053a90602f2f32ad461", install = TRUE, overwrite = TRUE)
readRenviron("~/.Renviron")

vars <- tidycensus::load_variables(year = 2020, dataset = "dhc")
vars <- vars %>%
    dplyr::filter(stringr::str_detect(concept, "RACE"))

# When getting data from the decennial census by "tract" or "block group", the state
# must be specified

clean_race <- function(df) {
    df %>%
        tidyr::pivot_wider(
            id_cols = GEOID, names_from = variable, values_from = value
        ) %>%
        dplyr::rename(
            total = P11_001N,
            hispanic = P11_002N,
            nh_total = P11_003N,
            nh_white = P11_005N,
            nh_black = P11_006N,
            nh_aian = P11_007N,
            nh_asian = P11_008N,
            nh_hpi = P11_009N,
            nh_other = P11_010N,
            nh_multiracial = P11_011N,
            nh_white_other = P11_017N,
            nh_black_other = P11_021N,
            nh_aian_other = P11_024N,
            nh_asian_other = P11_026N,
            nh_hpi_other = P11_027N,
            nh_asian_hpi_other = P11_048N,
        ) %>%
        dplyr::select(
            GEOID, total, hispanic, nh_total,
            nh_white, nh_black, nh_aian, nh_hpi, nh_other,
            nh_multiracial, nh_white_other, nh_black_other,
            nh_aian_other, nh_asian_other, nh_hpi_other,
            nh_asian_hpi_other
        )
}

get_race_by_zcta <- function() {
    df <- tidycensus::get_decennial(
        geography = "zcta",
        table = "P11",
        year = 2020,
        sumfile = "dhc",
        cache_table = TRUE
    )
    arrow::write_parquet(df, file.path(CENSUS_PATH, "zcta/raw/zcta.parquet"))
}

get_race_by_tract <- function() {
    for (state in unique(tidycensus::fips_codes$state_code)) {
        try({
            df <- tidycensus::get_decennial(
                geography = "tract",
                state = state,
                table = "P11",
                year = 2020,
                sumfile = "dhc",
            )

            arrow::write_parquet(
                df,
                file.path(
                    CENSUS_PATH, stringr::str_glue("tract/raw/{state}.parquet")
                )
            )
        })
    }
}

get_race_by_block_group <- function() {
    for (state in unique(tidycensus::fips_codes$state_code)) {
        try({
            df <- tidycensus::get_decennial(
                geography = "block group",
                state = state,
                table = "P11",
                year = 2020,
                sumfile = "dhc",
            )

            arrow::write_parquet(
                df,
                file.path(
                    CENSUS_PATH, stringr::str_glue("block_group/raw/{state}.parquet")
                )
            )
        })
    }
}


get_race_by_zcta()
# get_race_by_tract()
# get_race_by_block_group()



# get_race_by_zcta <- function(year) {
#     race_by_zcta <- tidycensus::get_acs(geography = "zcta", table = "B03002", year = year, cache_table = TRUE) %>%
#         tidyr::pivot_wider(id_cols = NAME, names_from = variable, values_from = estimate) %>%
#         dplyr::rename(
#             zcta = NAME,
#             Total_Pop = B03002_001,
#             Non_Hispanic_Total = B03002_002,
#             NH_White_alone = B03002_003,
#             NH_AIAN_alone = B03002_005,
#             NH_Black_alone = B03002_004,
#             nh_asian = B03002_006,
#             nh_hawaiian = B03002_007,
#             other = B03002_008,
#             multiple = B03002_009,
#             Hispanic_Total = B03002_012,
#         ) %>%
#         dplyr::mutate(
#             zcta = zcta %>%
#                 stringr::str_remove("ZCTA5") %>%
#                 stringr::str_trim(),
#             year = year,
#             nh_api = nh_asian + nh_hawaiian,
#         ) %>%
#         dplyr::select(zcta, nh_white, nh_black, nh_native, nh_api, hispanic, multiple)

#     return(race_by_zcta)
# }

# race_by_zcta <- get_race_by_zcta(2021)
# arrow::write_parquet(race_by_zcta, file.path(CENSUS_PATH, "acs5yr2021_race.parquet"))
