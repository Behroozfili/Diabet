"""
Data preparation module for Diabetes Prediction project.

This module provides functions to load raw data, handle missing values,
and perform feature scaling using scikit-learn's StandardScaler.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config


def load_data(filepath):
    """
    Load data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        data = pd.read_csv(filepath)
        if data.empty:
            raise ValueError("Loaded CSV file is empty.")
        return data
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")


def handle_missing_values(data, strategy="drop"):
    """
    Handle missing values in the dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        strategy (str): Strategy to handle missing values.
                       Options: 'drop' (default), 'mean', 'median', 'ffill', 'bfill'.

    Returns:
        pd.DataFrame: Dataset with missing values handled.

    Raises:
        ValueError: If strategy is not recognized.
    """
    valid_strategies = ["drop", "mean", "median", "ffill", "bfill"]
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")

    data_copy = data.copy()

    if strategy == "drop":
        data_copy = data_copy.dropna()
    elif strategy == "mean":
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        data_copy[numeric_cols] = data_copy[numeric_cols].fillna(data_copy[numeric_cols].mean())
    elif strategy == "median":
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        data_copy[numeric_cols] = data_copy[numeric_cols].fillna(data_copy[numeric_cols].median())
    elif strategy == "ffill":
        data_copy = data_copy.fillna(method="ffill")
    elif strategy == "bfill":
        data_copy = data_copy.fillna(method="bfill")

    return data_copy


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        X_test (pd.DataFrame or np.ndarray): Testing features.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
               - X_train_scaled: Scaled training features.
               - X_test_scaled: Scaled testing features.
               - scaler: Fitted StandardScaler object for future transformations.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(filepath, target_column, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    Complete preprocessing pipeline: load, handle missing values, split, and scale.

    Args:
        filepath (str): Path to the raw CSV file.
        target_column (str): Name of the target column.
        test_size (float): Proportion of data to use for testing (default: config.TEST_SIZE).
        random_state (int): Random seed for reproducibility (default: config.RANDOM_STATE).

    Returns:
        dict: Dictionary containing:
              - X_train: Training features (scaled).
              - X_test: Testing features (scaled).
              - y_train: Training labels.
              - y_test: Testing labels.
              - scaler: Fitted StandardScaler object.
    """
    # Load data
    data = load_data(filepath)

    # Handle missing values
    data = handle_missing_values(data, strategy="drop")

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    # Example usage (assuming diabetes.csv exists in data/raw/)
    sample_file = os.path.join(config.DATA_PATHS["raw"], "diabetes.csv")
    if os.path.exists(sample_file):
        preprocessed = preprocess_data(sample_file, target_column="Outcome")
        print("Preprocessing completed successfully!")
        print(f"Training set shape: {preprocessed['X_train'].shape}")
        print(f"Testing set shape: {preprocessed['X_test'].shape}")
    else:
        print(f"Sample data file not found at {sample_file}")
