"""Train, compare, tune and evaluate rainfall classifiers.

Usage:
    python -m rainfall.train --data data/Rainfall.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import loguniform  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

from .data import RAW_FEATURES, TARGET, load_data  # noqa: E402
from .features import engineer  # noqa: E402

SEED = 42


def build_pipeline(clf) -> Pipeline:
    """features -> impute -> scale -> classifier (all fitted on train data only)."""
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def candidates() -> dict[str, Pipeline]:
    return {
        "Baseline (majority class)": build_pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
            )
        ),
        "Gradient Boosting": build_pipeline(GradientBoostingClassifier(random_state=SEED)),
    }


SEARCH_SPACES = {
    "Logistic Regression": {"clf__C": loguniform(1e-3, 1e2)},
    "Random Forest": {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 4, 6, 8, 12],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__max_depth": [1, 2, 3],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__min_samples_leaf": [1, 3, 5, 10],
    },
}


def compare_models(X_train, y_train) -> pd.DataFrame:
    """Repeated stratified CV on the training split only."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "balanced_accuracy": "balanced_accuracy"}
    rows = []
    for name, pipe in candidates().items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "roc_auc": res["test_roc_auc"].mean(),
                "roc_auc_std": res["test_roc_auc"].std(),
                "f1": res["test_f1"].mean(),
                "balanced_accuracy": res["test_balanced_accuracy"].mean(),
            }
        )
        print(f"  {name:<28} AUC {rows[-1]['roc_auc']:.3f} +/- {rows[-1]['roc_auc_std']:.3f}")
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def tune(name: str, X_train, y_train):
    space = SEARCH_SPACES[name]
    search = RandomizedSearchCV(
        candidates()[name],
        space,
        n_iter=40,
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def bootstrap_auc_ci(y_true, y_prob, n=1000, seed=SEED) -> tuple[float, float]:
    """95% bootstrap CI: with a small test set a single AUC number is noisy."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate(model, X_test, y_test) -> dict:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    lo, hi = bootstrap_auc_ci(y_test, prob)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "roc_auc_ci95": [lo, hi],
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_test": int(len(y_test)),
        "report": classification_report(
            y_test, pred, target_names=["No rain", "Rain"], zero_division=0
        ),
    }


# ---------------------------------------------------------------- plots
def save_plots(df, comparison, model, X_test, y_test, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # class balance + correlations
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    counts = df[TARGET].map({1: "Rain", 0: "No rain"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax[0], hue=counts.index, legend=False)
    ax[0].set_title("Class balance")
    ax[0].set_ylabel("days")
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1],
                annot_kws={"size": 7}, cbar=False)
    ax[1].set_title("Correlation matrix (raw features)")
    fig.tight_layout()
    fig.savefig(out / "eda.png", dpi=150)
    plt.close(fig)

    # model comparison
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(comparison["model"][::-1], comparison["roc_auc"][::-1],
            xerr=comparison["roc_auc_std"][::-1], color="#4C78A8")
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("ROC-AUC (5-fold x 3 repeats, training split)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    # confusion matrix + ROC on held-out test
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No rain", "Rain"], cmap="Blues",
        colorbar=False, ax=ax[0])
    ax[0].set_title("Confusion matrix (test set)")
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1], name="Tuned model")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(out / "test_performance.png", dpi=150)
    plt.close(fig)

    # permutation importance on RAW columns (what the user actually enters)
    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                 n_repeats=30, random_state=SEED, n_jobs=-1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(X_test.columns)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#F58518")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to Rainfall CSV")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args.data)
    print(f"Loaded {df.shape[0]} rows | rain rate {df[TARGET].mean():.1%} | "
          f"missing cells: {int(df.isna().sum().sum())}")

    X, y = df[RAW_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=SEED)

    print("\n[1/3] Cross-validated model comparison")
    comparison = compare_models(X_train, y_train)
    best_name = comparison.loc[
        comparison["model"] != "Baseline (majority class)", "model"].iloc[0]

    print(f"\n[2/3] Tuning {best_name}")
    search = tune(best_name, X_train, y_train)
    best = search.best_estimator_
    print(f"  best CV ROC-AUC {search.best_score_:.3f} | params {search.best_params_}")

    print("\n[3/3] Held-out test evaluation")
    metrics = evaluate(best, X_test, y_test)
    print(metrics["report"])
    print(f"  ROC-AUC {metrics['roc_auc']:.3f} "
          f"(95% CI {metrics['roc_auc_ci95'][0]:.3f}-{metrics['roc_auc_ci95'][1]:.3f})")

    reports = Path(args.reports_dir)
    save_plots(df, comparison, best, X_test, y_test, reports)
    comparison.to_csv(reports / "model_comparison.csv", index=False)

    # Deployed model: same hyper-parameters, refit on ALL data (metrics above stay honest
    # because they came from the train-only model).
    final = clone(best).fit(X, y)
    models = Path(args.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, models / "rainfall_model.joblib")

    metadata = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "model": best_name,
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "cv_roc_auc_tuned": float(search.best_score_),
        "test_metrics": {k: v for k, v in metrics.items() if k != "report"},
        "n_rows": int(len(df)),
        "rain_rate": float(y.mean()),
        "feature_ranges": {
            c: {"min": float(X[c].min()), "max": float(X[c].max()),
                "median": float(X[c].median())} for c in RAW_FEATURES},
        "class_medians": {
            "rain": X[y == 1].median().to_dict(),
            "no_rain": X[y == 0].median().to_dict(),
        },
    }
    (models / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model -> {models/'rainfall_model.joblib'} | plots -> {reports}/")


if __name__ == "__main__":
    main()
