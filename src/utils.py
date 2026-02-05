"""
Utility functions for Diabetes Prediction project.

This module provides utility functions for saving and loading Python objects
(models, scalers, etc.) using joblib.
"""

import os
import joblib
import config


def save_model(model, model_name="model.pkl"):
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained model object.
        model_name (str): Name of the model file (default: "model.pkl").

    Returns:
        str: Path to the saved model file.

    Raises:
        OSError: If the model cannot be saved.
    """
    model_path = os.path.join(config.MODELS_DIR, model_name)

    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved successfully to: {model_path}")
        return model_path
    except OSError as e:
        raise OSError(f"Failed to save model: {e}")


def load_model(model_name="model.pkl"):
    """
    Load a trained model from disk using joblib.

    Args:
        model_name (str): Name of the model file (default: "model.pkl").

    Returns:
        object: Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If the model cannot be loaded.
    """
    model_path = os.path.join(config.MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")


def save_scaler(scaler, scaler_name="scaler.pkl"):
    """
    Save a fitted scaler to disk using joblib.

    Args:
        scaler: Fitted StandardScaler object.
        scaler_name (str): Name of the scaler file (default: "scaler.pkl").

    Returns:
        str: Path to the saved scaler file.

    Raises:
        OSError: If the scaler cannot be saved.
    """
    scaler_path = os.path.join(config.MODELS_DIR, scaler_name)

    try:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved successfully to: {scaler_path}")
        return scaler_path
    except OSError as e:
        raise OSError(f"Failed to save scaler: {e}")


def load_scaler(scaler_name="scaler.pkl"):
    """
    Load a fitted scaler from disk using joblib.

    Args:
        scaler_name (str): Name of the scaler file (default: "scaler.pkl").

    Returns:
        object: Loaded scaler object.

    Raises:
        FileNotFoundError: If the scaler file does not exist.
        Exception: If the scaler cannot be loaded.
    """
    scaler_path = os.path.join(config.MODELS_DIR, scaler_name)

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from: {scaler_path}")
        return scaler
    except Exception as e:
        raise Exception(f"Failed to load scaler: {e}")


def save_artifact(artifact, artifact_name, artifact_dir=None):
    """
    Generic utility function to save any Python object using joblib.

    Args:
        artifact: Object to save.
        artifact_name (str): Name of the artifact file.
        artifact_dir (str): Directory to save to (default: config.MODELS_DIR).

    Returns:
        str: Path to the saved artifact file.

    Raises:
        OSError: If the artifact cannot be saved.
    """
    if artifact_dir is None:
        artifact_dir = config.MODELS_DIR

    artifact_path = os.path.join(artifact_dir, artifact_name)

    try:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(artifact, artifact_path)
        print(f"Artifact saved successfully to: {artifact_path}")
        return artifact_path
    except OSError as e:
        raise OSError(f"Failed to save artifact: {e}")


def load_artifact(artifact_name, artifact_dir=None):
    """
    Generic utility function to load any Python object using joblib.

    Args:
        artifact_name (str): Name of the artifact file.
        artifact_dir (str): Directory to load from (default: config.MODELS_DIR).

    Returns:
        object: Loaded artifact object.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
        Exception: If the artifact cannot be loaded.
    """
    if artifact_dir is None:
        artifact_dir = config.MODELS_DIR

    artifact_path = os.path.join(artifact_dir, artifact_name)

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    try:
        artifact = joblib.load(artifact_path)
        print(f"Artifact loaded successfully from: {artifact_path}")
        return artifact
    except Exception as e:
        raise Exception(f"Failed to load artifact: {e}")


if __name__ == "__main__":
    # Example usage (requires trained model and scaler)
    print("Utility module loaded successfully!")
    print("Use this module to save and load models, scalers, and other artifacts.")
