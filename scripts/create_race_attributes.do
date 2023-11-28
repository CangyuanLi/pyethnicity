/*
This script is taken from https://github.com/cfpb/proxy-methodology/blob/master/scripts/create_attr_over18_all_geo_entities.do, which is the CFPB's Github repo for their BISG methodology. Some edits are made to use my filenames and variable names.
*/


* This script uses the base information from the census flat files for block group, tract, and ZIP code and allocates "Some Other Race"
* to each group in proportion.  It creates three files (one each for block group, tract, and ZIP code) containing the geography-only
* proxy as well as the proportion of population for a given race and ethnicity residing in a given geographic area, which is 
* used to build the BISG proxy.   

set more off
set type double

* Set input and output directories.
local indir = "./input_files"
local outdir = "./created_files"

* Set names for geo entity files.
local geo_files = "block_group tract zcta"

foreach file in `geo_files'{
	use "`indir'/race_by_`file'_2020.dta", clear
    gen file = "`file'"
    
* Step 1: From the SF1, retain population counts for the contiguous U.S., Alaska, and Hawaii in order to ensure consistency with the population
* covered by the census surname list.
	if "`file'" != "zcta" {
		drop if state_fips == "72"
	}
    
    if "`file'" == "zcta" {
		drop if inlist(substr(zcta,1,3),"006","007","008","009")
	}
		
	rename nh_white NH_White_alone
	rename nh_white_other NH_White_Other
	rename nh_black NH_Black_alone
	rename nh_black_other NH_Black_Other
	rename nh_aian NH_AIAN_alone
	rename nh_aian_other NH_AIAN_Other
	rename nh_asian NH_Asian_alone
	rename nh_asian_other NH_Asian_Other
	rename nh_hpi NH_HPI_alone
	rename nh_hpi_other NH_HPI_Other
	rename nh_asian_hpi NH_Asian_HPI
	rename nh_asian_hpi_other NH_Asian_HPI_Other
	rename nh_multiracial NH_Mult_Total
	rename total Total_Pop
	rename hispanic Hispanic_Total
	rename nh_other NH_Other_alone
	rename nh_total Non_Hispanic_Total
	
	generate NH_API_alone = NH_Asian_alone + NH_HPI_alone
	generate NH_API_Other = NH_Asian_Other + NH_HPI_Other
	

* Step 2: Address "Other" category from 2010 Census; what is done here follows Word(2008).
    foreach x in NH_White NH_Black NH_AIAN NH_API {
	  	replace `x'_alone = `x'_alone + `x'_Other
	  }

* Census breaks out Asian and PI separately; since we consider them as one, we correct for this.
    replace NH_API_alone = NH_API_alone + NH_Asian_HPI + NH_Asian_HPI_Other

* Replace multiracial total to account for the fact that we have suppressed the Other category.
    replace NH_Mult_Total = NH_Mult_Total - (NH_White_Other + NH_Black_Other + NH_AIAN_Other + NH_Asian_HPI + NH_API_Other + NH_Asian_HPI_Other)

* Verify the steps above by confirming that the Total Population still matches.

    assert Total_Pop == (NH_White_alone + NH_Black_alone + NH_API_alone + NH_AIAN_alone + NH_Mult_Total + NH_Other_alone + Hispanic_Total)

* Step 3: Proportionally redistribute Non-Hispanic Other population to remaining Non-Hispanic groups within each block.
    foreach x in NH_White_alone NH_Black_alone NH_AIAN_alone NH_API_alone NH_Mult_Total {
	   replace `x' = `x' + (`x' / (Total_Pop-Hispanic_Total-NH_Other_alone)) * NH_Other_alone
	   replace `x' = 0 if Total_Pop == 0
	   replace `x' = NH_Other_alone / 5 if Non_Hispanic_Total == NH_Other_alone
	  }
	
	   egen pop_check = rowtotal(NH_White_alone NH_Black_alone NH_AIAN_alone NH_API_alone NH_Mult_Total Hispanic_Total)

    assert round(pop_check,1) == Total_Pop

    preserve

* Collapse dataset to get Population Totals for each group.
    collapse (sum) NH_White_alone NH_Black_alone NH_AIAN_alone NH_API_alone NH_Mult_Total Hispanic_Total Total_Pop

    local national_pop = Total_Pop
    local national_nh_white_alone = NH_White_alone
    local national_nh_black_alone = NH_Black_alone
    local national_nh_aian_alone = NH_AIAN_alone
    local national_nh_asian_alone = NH_API_alone
    local national_nh_hawn_alone = 0
    local national_hispanic_pop = Hispanic_Total
* End constants.

    restore

    gen geo_pr_white = NH_White_alone / Total_Pop
    gen geo_pr_black = NH_Black_alone / Total_Pop
    gen geo_pr_aian = NH_AIAN_alone / Total_Pop
    gen geo_pr_api = NH_API_alone / Total_Pop

* Multiple races or "some other race" (and not Hispanic).
    gen geo_pr_mult_other = (NH_Mult_Total) / Total_Pop
    gen geo_pr_hispanic = Hispanic_Total / Total_Pop

* When updating geocoded race probabilities, we require the probability that someone of a particular race lives in that block group, tract, or ZIP code. 
* Our race counts are single race reported counts, therefore we divide the single race population within each block by the total single race population
* for each group.

    local national_nh_mult_other = `national_pop' - `national_hispanic_pop' - `national_nh_white_alone' - `national_nh_black_alone' - `national_nh_aian_alone' - `national_nh_asian_alone' - `national_nh_hawn_alone'
    n di "Number of other-race or multiple-race non-Hispanics: `national_nh_mult_other'"

    local national_nh_api_alone = `national_nh_asian_alone' + `national_nh_hawn_alone'

    gen here = Total_Pop / `national_pop'
    gen here_given_white = NH_White_alone / `national_nh_white_alone'
    gen here_given_black = NH_Black_alone / `national_nh_black_alone'
    gen here_given_aian = NH_AIAN_alone / `national_nh_aian_alone'
    gen here_given_api = NH_API_alone / `national_nh_api_alone'
    gen here_given_mult_other = (NH_Mult_Total) / `national_nh_mult_other'
    gen here_given_hispanic = Hispanic_Total / `national_hispanic_pop'

    if "`file'" == "block_group"{
    	keep block_group geo_pr* here*
    }
    if "`file'" == "tract"{
    	keep tract geo_pr* here*
    }
    if "`file'"=="zcta"{
    	keep zcta geo_pr* here*
    }     
	
    compress
	save "`outdir'/race_by_`file'_2020.dta", replace
}

exit

*END