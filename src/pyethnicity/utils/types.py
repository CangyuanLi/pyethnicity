from __future__ import annotations

from typing import Literal, Union

from polars.series.series import ArrayLike

Name = Union[str, ArrayLike]
Geography = Union[int, str, ArrayLike]
GeoType = Literal["zcta", "tract", "block_group"]
Model = Literal["first_last", "first_sex"]
Year = Union[int, ArrayLike]
