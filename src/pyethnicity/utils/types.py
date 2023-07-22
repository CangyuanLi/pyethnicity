from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow

ArrayLike = Union[Sequence, pd.Series, pl.Series, np.ndarray, pyarrow.Array]
Model = Literal["bilstm", "dual_bilstm"]
