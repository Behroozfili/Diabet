"""
Model factory module for Diabetes Prediction project.

This module provides a factory function to initialize and configure
the RandomForestClassifier with parameters from the config module.
"""

from sklearn.ensemble import RandomForestClassifier
import config


def create_model():
    """
    Initialize and return a RandomForestClassifier with parameters from config.

    The model is configured with the following hyperparameters:
    - n_estimators: Number of trees in the forest (from config.N_ESTIMATORS)
    - max_depth: Maximum depth of each tree (from config.MAX_DEPTH)
    - random_state: Seed for reproducibility (from config.RANDOM_STATE)

    Returns:
        sklearn.ensemble.RandomForestClassifier: Configured RandomForestClassifier instance.
    """
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,  # Use all available processors
        verbose=0,
    )
    return model


if __name__ == "__main__":
    # Example usage
    model = create_model()
    print("Model created successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model parameters:")
    print(f"  - n_estimators: {model.n_estimators}")
    print(f"  - max_depth: {model.max_depth}")
    print(f"  - random_state: {model.random_state}")
