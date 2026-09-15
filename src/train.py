"""Train and evaluate beginner-friendly flood prediction baseline models.

The script never alters the source CSV. It writes reports, charts, metrics, and
the best trained model to the project output folders.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Avoid relying on a user-profile directory that may not be writable.
_matplotlib_cache = Path(__file__).resolve().parents[1] / ".matplotlib"
_matplotlib_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
import matplotlib
matplotlib.use("Agg")  # Enables plot creation without opening GUI windows.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
TARGET = "Flood Occurred"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flood prediction baselines.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to source CSV. Defaults to the only non-empty CSV in data/.",
    )
    return parser.parse_args()


def find_data_file(requested: Path | None) -> Path:
    """Find a non-empty CSV without assuming a particular dataset filename."""
    if requested:
        path = requested if requested.is_absolute() else ROOT / requested
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"CSV is missing or empty: {path}")
        return path
    candidates = [p for p in (ROOT / "data").glob("*.csv") if p.stat().st_size > 0]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Pass --data PATH. Expected exactly one non-empty CSV in data/."
        )
    return candidates[0]


def make_output_dirs() -> tuple[Path, Path]:
    outputs, models = ROOT / "outputs", ROOT / "models"
    (outputs / "plots").mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return outputs, models


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Only normalize whitespace; do not rename semantic column labels."""
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def inspect_data(df: pd.DataFrame, target: str, outputs: Path) -> tuple[list[str], list[str]]:
    """Save a transparent data-quality report before modeling."""
    if target not in df.columns:
        raise ValueError(f"Required target column '{target}' was not found. Columns: {list(df.columns)}")
    if df[target].isna().any():
        raise ValueError("The target column contains missing values and cannot be used for supervised learning.")
    target_values = set(pd.to_numeric(df[target], errors="coerce").dropna().unique())
    if target_values != {0, 1}:
        raise ValueError(f"Target must contain exactly 0 and 1; found {sorted(target_values)}")

    feature_frame = df.drop(columns=target)
    numeric = feature_frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in feature_frame.columns if column not in numeric]

    numeric_summary = df[numeric].describe().T if numeric else pd.DataFrame()
    iqr_flags: dict[str, int] = {}
    for column in numeric:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_flags[column] = int(((df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)).sum())

    report = {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numerical_columns": numeric,
        "categorical_columns": categorical,
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {str(key): int(value) for key, value in df[target].value_counts().sort_index().items()},
        "unique_values": {column: int(df[column].nunique(dropna=True)) for column in df.columns},
        "iqr_outlier_flags": iqr_flags,
        "notes": [
            "IQR flags identify values that are statistically unusual, not necessarily errors.",
            "Source CSV is read only; no rows or values are modified by this script.",
        ],
    }
    (outputs / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    numeric_summary.to_csv(outputs / "descriptive_statistics.csv")
    return numeric, categorical


def save_eda(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str], outputs: Path) -> None:
    """Create compact EDA figures and relationship plots."""
    plot_dir = outputs / "plots"
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=target, hue=target, legend=False, palette="Set2")
    ax.set_title("Flood occurrence distribution")
    for bar in ax.patches:
        ax.annotate(str(int(bar.get_height())), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(plot_dir / "target_distribution.png", dpi=160); plt.close()

    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing.index, y=missing.values, color="#4c78a8")
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Missing values"); plt.title("Missing-value analysis")
    plt.tight_layout(); plt.savefig(plot_dir / "missing_values.png", dpi=160); plt.close()

    if numeric:
        corr = df[numeric + [target]].corr(numeric_only=True)
        corr.to_csv(outputs / "correlation_matrix.csv")
        plt.figure(figsize=(max(9, len(corr) * 0.8), max(7, len(corr) * 0.7)))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
        plt.title("Numerical-feature correlation matrix")
        plt.tight_layout(); plt.savefig(plot_dir / "correlation_heatmap.png", dpi=160); plt.close()

        cols = 3
        rows = int(np.ceil(len(numeric) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        for ax, column in zip(np.ravel(axes), numeric):
            sns.histplot(data=df, x=column, hue=target, bins=30, kde=True, stat="density", common_norm=False, ax=ax)
            ax.set_title(column)
        for ax in np.ravel(axes)[len(numeric):]: ax.remove()
        fig.suptitle("Numerical feature distributions by flood outcome", y=1.01)
        fig.tight_layout(); fig.savefig(plot_dir / "numerical_distributions.png", dpi=160, bbox_inches="tight"); plt.close(fig)

        target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
        relationship_numeric = target_corr.head(min(6, len(target_corr))).index.tolist()
        fig, axes = plt.subplots(len(relationship_numeric), 1, figsize=(10, 4 * len(relationship_numeric)))
        axes = np.atleast_1d(axes)
        for ax, column in zip(axes, relationship_numeric):
            sns.boxplot(data=df, x=target, y=column, hue=target, legend=False, palette="Set2", ax=ax)
            ax.set_title(f"{column} by flood outcome")
        fig.tight_layout(); fig.savefig(plot_dir / "top_numeric_relationships.png", dpi=160); plt.close(fig)

    for column in categorical:
        plt.figure(figsize=(9, 5))
        sns.countplot(data=df, x=column, hue=target, palette="Set2")
        plt.xticks(rotation=30, ha="right"); plt.title(f"{column} by flood outcome")
        plt.tight_layout(); plt.savefig(plot_dir / f"categorical_{column.lower().replace(' ', '_')}.png", dpi=160); plt.close()


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_model(name: str, pipeline: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, outputs: Path) -> dict[str, float | str]:
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    cm = confusion_matrix(y_test, predictions)
    pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(outputs / f"confusion_matrix_{name.lower().replace(' ', '_')}.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Flood (0)", "Flood (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} confusion matrix")
    fig.tight_layout(); fig.savefig(outputs / "plots" / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160); plt.close(fig)
    return metrics


def save_random_forest_importance(model: Pipeline, outputs: Path) -> None:
    preprocessor = model.named_steps["preprocessor"]
    forest = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": forest.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(outputs / "random_forest_feature_importance.csv", index=False)
    top10 = importance.head(10)
    top10.to_csv(outputs / "random_forest_top_10_features.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10, x="importance", y="feature", color="#4c78a8")
    plt.title("Random Forest: top 10 feature importances")
    plt.tight_layout(); plt.savefig(outputs / "plots" / "random_forest_top_10_features.png", dpi=160); plt.close()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    data_path = find_data_file(args.data)
    outputs, models_dir = make_output_dirs()
    df = clean_column_names(pd.read_csv(data_path))
    numeric, categorical = inspect_data(df, TARGET, outputs)
    save_eda(df, TARGET, numeric, categorical, outputs)

    x, y = df.drop(columns=TARGET), df[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        # One worker is portable in restricted Windows environments; it does not affect reproducibility.
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=1),
    }
    all_metrics, fitted = [], {}
    for name, estimator in model_specs.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(numeric, categorical)), ("model", estimator)])
        all_metrics.append(evaluate_model(name, pipeline, x_train, x_test, y_train, y_test, outputs))
        fitted[name] = pipeline

    comparison = pd.DataFrame(all_metrics).sort_values(["roc_auc", "f1_score"], ascending=False)
    comparison.to_csv(outputs / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    joblib.dump(fitted[best_name], models_dir / "best_flood_model.joblib")
    save_random_forest_importance(fitted["Random Forest"], outputs)
    (outputs / "run_summary.json").write_text(json.dumps({"source_csv": str(data_path.relative_to(ROOT)), "best_model": best_name, "train_rows": len(x_train), "test_rows": len(x_test), "random_state": RANDOM_STATE, "test_size": TEST_SIZE}, indent=2), encoding="utf-8")
    print(f"Completed. Best model: {best_name}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
