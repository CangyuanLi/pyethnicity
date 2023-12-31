from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
import pandas as pd
import polars as pl

ArrayLike = Union[Sequence, pd.Series, pl.Series, np.ndarray]
Name = Union[str, ArrayLike]
Geography = Union[int, str, ArrayLike]
GeoType = Literal["zcta", "tract", "block_group"]
Model = Literal["first_last", "first_sex"]
Year = Union[int, ArrayLike]
