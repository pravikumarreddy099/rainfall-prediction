# 🌧️ Rainfall Prediction System

End-to-end machine-learning project that predicts whether it rained on a given day from daily weather
observations (pressure, temperature, dew point, humidity, cloud cover, sunshine, wind), with a
Streamlit app for single predictions, batch scoring and model insights.

**Stack:** Python · Pandas · NumPy · Scikit-learn · Matplotlib/Seaborn · Streamlit · Joblib · Pytest

## What makes this more than a notebook

| Concern | Approach |
|---|---|
| Data leakage | Imputation, scaling and feature engineering live **inside one sklearn `Pipeline`**, fitted on training folds only |
| Small, imbalanced data (366 days, ~68% rain) | Stratified split + `class_weight="balanced"` instead of throwing away 36% of rows by down-sampling |
| Model selection | Baseline, Logistic Regression, Random Forest, Gradient Boosting compared with **5-fold x 3-repeat stratified CV** on ROC-AUC / F1 / balanced accuracy |
| Tuning | `RandomizedSearchCV` on the winning model family |
| Honest evaluation | Test set touched once; **bootstrap 95% CI** on ROC-AUC because 74 test rows is noisy |
| Feature engineering | `temp_range`, `dew_depression` (replace collinear temperature columns), wind direction as sin/cos |
| Interpretability | Permutation importance on the raw inputs |
| Serving | Single saved pipeline (`joblib`) -> raw inputs in, probability out; same code path for the app and batch CSV |
| Quality | Unit tests, input sanity warnings in the UI, header cleaning for messy CSVs |

## Project layout

```
rainfall/            data.py (load/clean) · features.py (engineering) · train.py (compare, tune, evaluate, save)
app.py               Streamlit UI
data/                put Rainfall.csv here
models/              rainfall_model.joblib + metadata.json (created by training)
reports/             plots + model_comparison.csv (created by training)
tests/               pytest suite
scripts/             make_synthetic_data.py (smoke-testing only)
```

## Run it

```bash
pip install -r requirements.txt
python -m rainfall.train --data data/Rainfall.csv   # trains, evaluates, writes models/ and reports/
streamlit run app.py
pytest
```

## Results

Run training on the real data and paste your numbers here (they are printed and saved to
`models/metadata.json`):

| Model | CV ROC-AUC |
|---|---|
| Baseline | 0.50 |
| *(fill from reports/model_comparison.csv)* | |

Held-out test: ROC-AUC `__` (95% CI `__`–`__`), F1 `__`, recall `__`.

## Limitations (worth saying in an interview)

- One station, one year (366 days): results may not generalise to other places or seasons.
- Features are same-day observations, so this **classifies** whether it rained that day; it is not a
  next-day forecast. A forecasting version would use lagged features and a time-based split.
- With ~74 test rows, treat metrics as approximate; see the confidence interval.

## Ideas to extend

Lagged/rolling features with a time-series split · probability calibration · SHAP explanations ·
Docker + Streamlit Community Cloud deployment · GitHub Actions running pytest.

## Resume bullets (adapt with your real numbers)

- Built an end-to-end rainfall prediction system (Python, Pandas, NumPy, Scikit-learn, Streamlit) on 366 days of weather data, with a leakage-free preprocessing pipeline and engineered features (dew-point depression, temperature range, circular wind encoding).
- Benchmarked Logistic Regression, Random Forest and Gradient Boosting with repeated stratified cross-validation, tuned the best model, and reached **ROC-AUC X.XX (95% CI …)** on a held-out set; handled class imbalance with class weights rather than discarding data.
- Deployed an interactive Streamlit app with single-day prediction, batch CSV scoring, and model-insight dashboards; added unit tests and input validation.
