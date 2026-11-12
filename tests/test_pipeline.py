import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rainfall.data import RAW_FEATURES, TARGET, load_data
from rainfall.features import MODEL_FEATURES, engineer
from rainfall.train import build_pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "synth.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), str(p)],
                   check=True, capture_output=True)
    return p


def test_load_cleans_headers_and_target(csv_path):
    df = load_data(csv_path)
    assert list(df.columns) == RAW_FEATURES + [TARGET]
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df.isna().sum().sum() == 2  # kept as NaN for in-pipeline imputation


def test_engineer_shapes_and_wind_encoding():
    row = pd.DataFrame([dict.fromkeys(RAW_FEATURES, 10.0) | {"winddirection": 350.0, "maxtemp": 20.0}])
    out = engineer(row)
    assert list(out.columns) == MODEL_FEATURES
    assert out["temp_range"].iloc[0] == 10.0
    assert np.isclose(out["wind_sin"].iloc[0], np.sin(np.deg2rad(350)))


def test_pipeline_handles_nan_and_returns_probabilities(csv_path):
    df = load_data(csv_path)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(df[RAW_FEATURES], df[TARGET])
    prob = pipe.predict_proba(df[RAW_FEATURES])[:, 1]
    assert prob.shape == (len(df),) and ((prob >= 0) & (prob <= 1)).all()
