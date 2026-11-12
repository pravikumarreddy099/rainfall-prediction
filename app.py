"""Streamlit app: single prediction, batch scoring, model insights.

Run:  streamlit run app.py   (after `python -m rainfall.train --data data/Rainfall.csv`)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from rainfall.data import RAW_FEATURES, clean_columns

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "rainfall_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
REPORTS = ROOT / "reports"

# label, unit, step, help
FIELDS = {
    "pressure": ("Pressure", "hPa", 0.1, "Sea-level atmospheric pressure"),
    "temperature": ("Temperature", "°C", 0.1, "Mean temperature of the day"),
    "maxtemp": ("Max temperature", "°C", 0.1, "Daily high"),
    "mintemp": ("Min temperature", "°C", 0.1, "Daily low"),
    "dewpoint": ("Dew point", "°C", 0.1, "Close to temperature = near-saturated air"),
    "humidity": ("Humidity", "%", 1.0, "Relative humidity"),
    "cloud": ("Cloud cover", "%", 1.0, "Sky covered by cloud"),
    "sunshine": ("Sunshine", "hours", 0.1, "Bright sunshine duration"),
    "winddirection": ("Wind direction", "°", 10.0, "0-360, degrees from north"),
    "windspeed": ("Wind speed", "km/h", 0.1, "Mean wind speed"),
}

st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️", layout="wide")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


model, meta = load_artifacts()

st.title("🌧️ Rainfall Prediction System")
st.caption("Predicts whether it rained on a given day from daily weather observations.")

if model is None:
    st.error("No trained model found. Run `python -m rainfall.train --data data/Rainfall.csv` first.")
    st.stop()

tab_predict, tab_batch, tab_insights = st.tabs(["🔮 Predict", "📄 Batch (CSV)", "📊 Model insights"])

# ------------------------------------------------------------------ single
with tab_predict:
    preset = st.radio(
        "Start from",
        ["Dataset median", "Typical rainy day", "Typical dry day"],
        horizontal=True,
    )
    defaults = {
        "Dataset median": {c: meta["feature_ranges"][c]["median"] for c in RAW_FEATURES},
        "Typical rainy day": meta["class_medians"]["rain"],
        "Typical dry day": meta["class_medians"]["no_rain"],
    }[preset]

    values, cols = {}, st.columns(5)
    for i, name in enumerate(RAW_FEATURES):
        label, unit, step, help_ = FIELDS[name]
        rng = meta["feature_ranges"][name]
        pad = (rng["max"] - rng["min"]) * 0.25
        with cols[i % 5]:
            values[name] = st.number_input(
                f"{label} ({unit})",
                min_value=float(rng["min"] - pad),
                max_value=float(rng["max"] + pad),
                value=float(defaults[name]),
                step=step,
                help=help_,
                key=f"{name}-{preset}",  # reset widgets when the preset changes
            )

    problems = []
    if values["mintemp"] > values["maxtemp"]:
        problems.append("Min temperature is higher than max temperature.")
    if values["dewpoint"] > values["temperature"] + 1:
        problems.append("Dew point is above air temperature, which is physically unusual.")
    for p in problems:
        st.warning(p)

    if st.button("Predict", type="primary"):
        prob = float(model.predict_proba(pd.DataFrame([values])[RAW_FEATURES])[0, 1])
        left, right = st.columns([1, 2])
        left.metric("Probability of rain", f"{prob:.0%}")
        if prob >= 0.5:
            right.success("🌧️ Rain likely")
        else:
            right.info("☀️ Rain unlikely")
        right.progress(prob)
        st.caption(
            f"Model: {meta['model']} · test ROC-AUC "
            f"{meta['test_metrics']['roc_auc']:.2f}. This is a statistical estimate, "
            "not an official forecast."
        )

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Upload a CSV with these columns (header spelling/whitespace is cleaned automatically):")
    st.code(", ".join(RAW_FEATURES))
    st.download_button(
        "Download CSV template",
        pd.DataFrame(columns=RAW_FEATURES).to_csv(index=False),
        file_name="rainfall_template.csv",
    )
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = clean_columns(pd.read_csv(up))
        missing = [c for c in RAW_FEATURES if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            feats = raw[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
            out = raw.copy()
            out["rain_probability"] = model.predict_proba(feats)[:, 1].round(3)
            out["prediction"] = (out["rain_probability"] >= 0.5).map({True: "Rain", False: "No rain"})
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False),
                               file_name="rainfall_predictions.csv")

# ------------------------------------------------------------------ insights
with tab_insights:
    m = meta["test_metrics"]
    lo, hi = m["roc_auc_ci95"]
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{m['roc_auc']:.2f}", help=f"95% bootstrap CI {lo:.2f}-{hi:.2f}")
    c[1].metric("Accuracy", f"{m['accuracy']:.0%}")
    c[2].metric("Precision", f"{m['precision']:.0%}")
    c[3].metric("Recall", f"{m['recall']:.0%}")
    c[4].metric("F1", f"{m['f1']:.2f}")
    st.caption(
        f"Held-out test set of {m['n_test']} days · chosen model: **{meta['model']}** "
        f"· trained on {meta['n_rows']} days · ROC-AUC 95% CI {lo:.2f}-{hi:.2f}."
    )

    cmp_path = REPORTS / "model_comparison.csv"
    if cmp_path.exists():
        st.subheader("Models compared (cross-validation)")
        st.dataframe(pd.read_csv(cmp_path).round(3), use_container_width=True, hide_index=True)

    for img, title in [
        ("test_performance.png", "Test-set performance"),
        ("feature_importance.png", "What drives the prediction"),
        ("eda.png", "Data overview"),
    ]:
        if (REPORTS / img).exists():
            st.subheader(title)
            st.image(str(REPORTS / img))
