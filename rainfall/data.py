"""Loading and cleaning the raw weather CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "rainfall"

# Raw inputs the model needs (also what the Streamlit app asks the user for).
RAW_FEATURES = [
    "pressure",
    "maxtemp",
    "temperature",
    "mintemp",
    "dewpoint",
    "humidity",
    "cloud",
    "sunshine",
    "winddirection",
    "windspeed",
]

# The source CSV has a typo and stray whitespace in some headers.
_RENAMES = {"temparature": "temperature"}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase and fix known header typos."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.rename(columns=_RENAMES)


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the CSV and return RAW_FEATURES + binary TARGET (1 = rain, 0 = no rain).

    Missing feature values are kept as NaN on purpose: imputation happens inside
    the model pipeline so it is fitted on training folds only (no leakage).
    """
    df = clean_columns(pd.read_csv(path))
    df = df.drop(columns=["day"], errors="ignore")  # day-of-month: no signal

    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    target = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if target.isna().any():
        raise ValueError("`rainfall` must contain only 'yes' / 'no' values.")

    features = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    out = features.assign(**{TARGET: target.astype(int)})
    return out.reset_index(drop=True)
