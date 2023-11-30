# pyethnicity:
[![PyPI version](https://badge.fury.io/py/pyethnicity.svg)](https://badge.fury.io/py/pyethnicity)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyethnicity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://github.com/CangyuanLi/pyethnicity/actions/workflows/tests.yaml/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## What is it?

**pyethnicity** is a Python package to predict race from name and location and sex from first name. To the best of the author's knowledge, it outperforms all existing open-source models. It does this by training a Bidirectional LSTM on the largest, most comprehensive dataset of name and self-reported race thus far. It uses voter registration data from all 50 states. Additionally, it incorporates location features and improved versions of Bayesian Improved Surname Geocoding and Bayesian Improved Firstname Surname Geocoding to form an ensemble model that achieves up to 36.8% higher F1 scores than the next-best performing model. Finally, it provides CFPB-compliant and up-to-date versions of BISG and BIFSG.

# Usage:

Please see https://pyethnicity.readthedocs.io/en/latest/ for full documentation.

## Installing

The easiest way is to install **pyethnicity** is from PyPI using pip:

```sh
pip install pyethnicity
```

## Running

Pyethnicity exposes several functions. It supports block group, tract, and zip code level features. Each function takes in a scalar or array-like of inputs and returns a polars DataFrame of the input and the predictions.

```python
import pyethnicity

zcta = 27106
tract = 72153750502
first_name = "cangyuan"
last_name = "luo"

pyethnicity.bisg(last_name, zcta=zcta)
pyethnicity.bifsg(first_name, last_name, zcta=zcta, tract=tract)
pyethnicity.predict_race_fl(first_name, last_name)
pyethnicity.predict_race_flg(first_name, last_name, tract=tract)
pyethnicity.predict_race(first_name, last_name, zcta=zcta)
```


## Performance

**pyethnicity**

![](https://github.com/CangyuanLi/pyethnicity/raw/master/assets/ensemble_stats.png)

**rethnicity**

![](https://github.com/CangyuanLi/pyethnicity/raw/master/assets/reth_stats.png)

**ethnicolr**

![](https://github.com/CangyuanLi/pyethnicity/raw/master/assets/eth_stats.png)

Please see the correpsonding paper ["Can We Trust Race Prediction?"](https://github.com/CangyuanLi/pyethnicity/blob/master/paper.pdf) for more details.


# TODO:

- Re-train model to support Native American and Multiracial.

This package is still in active development. Please report any issues!