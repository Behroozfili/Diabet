"""
Trainer module for Diabetes Prediction project.

This module handles model training with MLflow experiment tracking.
It logs model parameters, performance metrics, and the trained model artifact.
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score
import config


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the RandomForestClassifier and return predictions and metrics.

    Args:
        model: RandomForestClassifier instance.
        X_train (np.ndarray): Scaled training features.
        y_train (pd.Series): Training labels.
        X_test (np.ndarray): Scaled testing features.
        y_test (pd.Series): Testing labels.

    Returns:
        dict: Dictionary containing:
              - model: Trained model.
              - y_pred: Predictions on test set.
              - metrics: Dictionary of evaluation metrics.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
    }

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def log_experiment(model, metrics, model_params):
    """
    Log model parameters, metrics, and the model artifact to MLflow.

    Args:
        model: Trained RandomForestClassifier instance.
        metrics (dict): Dictionary of metrics (accuracy, f1_score, precision).
        model_params (dict): Dictionary of model hyperparameters.

    Returns:
        str: The MLflow run ID.
    """
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(model_params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log the model using mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        print(f"\nExperiment logged with Run ID: {run_id}")

        return run_id


def setup_mlflow(experiment_name=config.MLFLOW_EXPERIMENT_NAME):
    """
    Set up MLflow experiment tracking.

    Args:
        experiment_name (str): Name of the experiment (default: config.MLFLOW_EXPERIMENT_NAME).

    Returns:
        str: The experiment ID.
    """
    # Set experiment name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    print(f"MLflow Experiment set to: {experiment_name}")

    return experiment_id


if __name__ == "__main__":
    # Example usage (requires data)
    print("Trainer module loaded successfully!")
    print("Use this module within the main pipeline for training and logging experiments.")
