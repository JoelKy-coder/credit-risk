"""Train and persist the credit risk model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import split_and_preprocess
from src.training import evaluate_model
from src.utils import PROJECT_ROOT, clean_data, load_raw_data, merge_economic_indicators


MODEL_METRIC_COLUMNS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC",
    "Log Loss",
]


def train_and_save_artifacts(
    processed_path: Path | None = None,
    models_dir: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the tuned LightGBM model and save model, pipeline, stats, and metrics."""
    processed_path = processed_path or PROJECT_ROOT / "data" / "processed" / "cleaned_loans.csv"
    models_dir = models_dir or PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, econ_df = load_raw_data()
    cleaned_df = clean_data(train_df)
    modeling_df = merge_economic_indicators(cleaned_df, econ_df)
    modeling_df.to_csv(processed_path, index=False)

    x_train, x_test, y_train, y_test, pipeline, training_stats = split_and_preprocess(modeling_df)

    param_grid = {
        "max_depth": [3, 5],
        "num_leaves": [15, 31],
        "learning_rate": [0.05, 0.1],
    }
    base_model = LGBMClassifier(
        n_estimators=50,
        random_state=random_state,
        class_weight="balanced",
        verbose=-1,
        n_jobs=1,
    )
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
    )
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, x_test, y_test)
    metric_report = {column: metrics[column] for column in MODEL_METRIC_COLUMNS}
    metric_report["Best Params"] = str(grid_search.best_params_)

    joblib.dump(best_model, models_dir / "best_model.pkl")
    joblib.dump(pipeline, models_dir / "pipeline.pkl")
    joblib.dump(training_stats, models_dir / "training_stats.pkl")
    pd.DataFrame([metric_report]).to_csv(models_dir / "model_metrics.csv", index=False)

    return {
        "best_params": grid_search.best_params_,
        "metrics": metric_report,
        "processed_path": str(processed_path),
        "models_dir": str(models_dir),
    }


if __name__ == "__main__":
    result = train_and_save_artifacts()
    print("Best parameters:", result["best_params"])
    print(pd.Series(result["metrics"]))
