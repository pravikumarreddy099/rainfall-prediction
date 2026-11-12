"""Feature engineering, kept inside the sklearn Pipeline so train/serve match."""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "pressure",
    "temperature",
    "dewpoint",
    "humidity",
    "cloud",
    "sunshine",
    "windspeed",
    "temp_range",      # maxtemp - mintemp: diurnal range, small on overcast/rainy days
    "dew_depression",  # temperature - dewpoint: ~0 means air is near saturation
    "wind_sin",        # wind direction is circular (350 deg is close to 10 deg),
    "wind_cos",        # so encode it as sin/cos instead of a raw 0-360 number
]


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Turn raw weather columns into model features.

    Highly collinear raw columns (maxtemp, mintemp) are replaced by the
    informative differences instead of being dropped outright.
    NaNs propagate and are imputed by the next pipeline step.
    """
    X = X.copy()
    X["temp_range"] = X["maxtemp"] - X["mintemp"]
    X["dew_depression"] = X["temperature"] - X["dewpoint"]
    radians = np.deg2rad(X["winddirection"])
    X["wind_sin"] = np.sin(radians)
    X["wind_cos"] = np.cos(radians)
    return X[MODEL_FEATURES]
