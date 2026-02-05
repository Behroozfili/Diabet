"""
Configuration module for Diabetes Prediction project.

This module centralizes all configuration parameters used across the project,
including model hyperparameters, data paths, and test parameters.
"""

import os
from pathlib import Path


# ============================================================================
# Project Root Directory
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()

# ============================================================================
# Data Paths
# ============================================================================
DATA_PATHS = {
    "raw": os.path.join(PROJECT_ROOT, "data", "raw"),
    "processed": os.path.join(PROJECT_ROOT, "data", "processed"),
}

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ============================================================================
# Model Hyperparameters
# ============================================================================
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42

# ============================================================================
# Data Processing Parameters
# ============================================================================
TEST_SIZE = 0.2

# ============================================================================
# MLflow Configuration
# ============================================================================
MLFLOW_EXPERIMENT_NAME = "diabetes-prediction"
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

# ============================================================================
# DagShub Configuration (Optional - if using DagShub)
# ============================================================================
 DAGSHUB_REPO_OWNER = "behrooz.filzadeh"
 DAGSHUB_REPO_NAME = "Diabet""
