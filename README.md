# Rainfall Prediction System
https://rainfall-prediction-c5mkdx66uxtd2vzlrmqugg.streamlit.app/

End-to-end machine-learning project that predicts whether it rained on a given day from daily weather
observations (pressure, temperature, dew point, humidity, cloud cover, sunshine, wind), with a
Streamlit app for single predictions, batch scoring and model insights.

**Stack:** Python · Pandas · NumPy · Scikit-learn · Matplotlib/Seaborn · Streamlit · Joblib · Pytest

## RESULTS

| Model | CV ROC-AUC |
|---|---|
| Baseline | 0.50 |
| Logistic Regression (chosen) | 0.853 |
| Random Forest | 0.837 |
| Gradient Boosting | 0.809 |

Held-out test: ROC-AUC 0.92 (95% CI 0.86–0.97), accuracy 81%, F1 0.85, recall 0.82.

