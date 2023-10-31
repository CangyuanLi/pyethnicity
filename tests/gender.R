library(gender)

df <- read.csv("tests/ssa_test_names.csv")
res <- gender::gender(df$first_name, years = c(1990, 2000), method = "demo")

write.csv(res, "tests/gender_r_package_results.csv")
