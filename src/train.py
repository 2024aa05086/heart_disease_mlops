"""Training script for the heart disease classifiers.

This script loads the dataset, performs preprocessing, evaluates multiple
models using cross‑validation, selects the best performing model based on
ROC‑AUC, trains it on the full training data, and saves the fitted model
and preprocessing pipeline to disk. If MLflow is installed and a
tracking server is configured, the script will log runs, parameters,
metrics, and artifacts for experiment tracking.
"""

from __future__ import annotations

import os
import joblib
from pathlib import Path
# Attempt to import mlflow for experiment tracking. If mlflow is not
# available (e.g. in offline environments), set mlflow to None so that
# training proceeds without logging. See assignment instructions for
# experiment tracking【321068608172419†L56-L106】.
try:
    import mlflow  # type: ignore
    import mlflow.sklearn  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
import typer

from . import data as data_utils
from . import model as model_utils

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_path: str = typer.Option(..., help="Path to the CSV dataset"),
    target_col: str = typer.Option("HeartDisease", help="Name of the target column"),
    output_dir: str = typer.Option("models", help="Directory to save models and artifacts"),
    tracking_uri: str = typer.Option(None, help="MLflow tracking URI. If provided, metrics and models are logged."),
    experiment_name: str = typer.Option("heart-disease-experiments", help="MLflow experiment name"),
) -> None:
    """Entrypoint for model training."""
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = data_utils.load_dataset(data_path)

    # Preprocess features
    X_processed, y, preprocessor = data_utils.preprocess_features(df, target_col)

    # Build candidate models
    models = model_utils.build_models()

    # Evaluate models using cross‑validation
    results = model_utils.evaluate_models(models, X_processed, y)

    # Determine best model by ROC‑AUC
    best_model_name = max(results, key=lambda m: results[m]["roc_auc"])
    best_model = models[best_model_name]

    # Optionally initialize MLflow. Only attempt to do so if mlflow is available
    # and a tracking URI is provided. If mlflow is None or no tracking_uri is
    # specified, experiment tracking is skipped entirely. This behaviour ensures
    # that training can run in environments without mlflow installed. See
    # MLflow documentation for more details on how to set up a tracking server【321068608172419†L56-L106】.
    use_mlflow = mlflow is not None and tracking_uri is not None
    if use_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.autolog(disable=False)
    import contextlib
    run_context = (
        mlflow.start_run(run_name=f"train_{best_model_name}")  # type: ignore
        if use_mlflow
        else contextlib.nullcontext()
    )
    # Open the MLflow run if tracking is enabled; otherwise use a no‑op context
    with run_context:
        # Log cross‑validation results if mlflow is enabled
        if use_mlflow:
            for model_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    # Prefix metric with model name to distinguish
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)  # type: ignore

        # Train final model on full data
        fitted_model = model_utils.train_final_model(best_model, X_processed, y)

        # Save the model and preprocessor to disk
        model_file = output_path / f"{best_model_name}_model.pkl"
        preproc_file = output_path / "preprocessor.pkl"
        joblib.dump(fitted_model, model_file)
        joblib.dump(preprocessor, preproc_file)

        # Log artifacts and parameters with MLflow if enabled
        if use_mlflow:
            mlflow.log_param("best_model", best_model_name)  # type: ignore
            mlflow.log_artifact(str(model_file))  # type: ignore
            mlflow.log_artifact(str(preproc_file))  # type: ignore

    # Save metrics to a JSON file
    import json
    metrics_file = output_path / "metrics.json"
    with metrics_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Training complete. Best model: {best_model_name}. Metrics saved to {metrics_file}.")


if __name__ == "__main__":
    app()  # pylint: disable=no-value-for-parameter