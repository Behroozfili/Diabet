"""
Diabetes Prediction Model - Source Package

This package contains all modules for the diabetes prediction ML pipeline.

Modules:
    - data_prep: Data loading and preprocessing
    - model_factory: Model initialization
    - trainer: Training and MLflow logging
    - utils: Utility functions for artifact persistence
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"
__description__ = "Diabetes Prediction using Random Forest Classifier with MLflow Tracking"

__all__ = ["data_prep", "model_factory", "trainer", "utils"]
