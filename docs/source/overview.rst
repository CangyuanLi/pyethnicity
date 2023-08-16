Overview
--------

**Pyethnicity** is a library that helps you proxy race / ethnicity based on name and
location. It is free and open-source, released under the MIT license. Race prediction
is important in many contexts. For example, the CFPB uses BISG to conduct their fair
lending analysis. Better race prediction can have real-world impact on lending,
healthcare, law, and more. This software is intended to serve as a tool for positive
and constructive purposes. By using this software, you agree to employ it in an
ethical manner. Under no circumstances should **pyethnicity** be used to engage in or
promote discrimination based on race, ethnicity, or any other characteristic, in any
shape or form.

The data used in this library can be found in its data folder. Geographic data (such
as the percent of Asian people in a ZCTA) come from the 2010 United States Census
`Summary file 1`_. Surname data is a combination of the 2010 United States Census
`Frequently Occurring Surnames`_ and proprietary 2022 voter registration data from 
L2, Inc. First name data is a combination of HMDA data sourced from "`Demographic
Aspects of First Names`_" [#]_ and the aforementioned voter registration data.

**Pyethnicity** provides routines for BISG, BIFSG, and a BiLSTM model. More details
about data, model development, and performance can be found in the corresponding paper,
"`Can We Trust Race Prediction?`_" [#]_.

.. _Summary File 1: https://www.census.gov/data/datasets/2010/dec/summary-file-1.html

.. _Frequently Occurring Surnames: https://www.census.gov/topics/population/genealogy/data/2010_surnames.html

.. _Demographic Aspects of First Names: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TYJKEZ

.. _Can We Trust Race Prediction?: https://arxiv.org/pdf/2307.08496.pdf

.. [#]
    Konstantinos Tzioumis, "Data for: Demographic aspects of first names".
    Harvard Dataverse (2018), V1 `<https://doi.org/10.7910/DVN/TYJKEZ>`_

.. [#]
    Cangyuan Li, "Can We Trust Race Prediction?".
    `<https://arxiv.org/pdf/2307.08496.pdf>`_