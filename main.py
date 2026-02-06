r"""
================================================================================
IMPORTANT: VIRTUAL ENVIRONMENT ACTIVATION REMINDER
================================================================================
Before running this script, activate the virtual environment:

Windows (PowerShell):
    .\Diabet\Scripts\Activate.ps1



Run the setup script (setup_env.ps1 on Windows or setup_env.sh on Mac/Linux)
to automatically create the virtual environment and install dependencies.
================================================================================

Main orchestrator module for Diabetes Prediction project.

This module serves as the central pipeline that integrates all components:
1. Data Loading and Preprocessing
2. Model Initialization
3. Model Training with MLflow Tracking
4. Model and Scaler Persistence

Execution flow:
    Load Data -> Preprocess -> Train -> Log Experiment -> Save Artifacts
"""

import os
import sys
import config
import dagshub
from src.data_prep import preprocess_data, save_processed_data
from src.model_factory import create_model
from src.trainer import train_model, setup_mlflow, log_experiment
from src.utils import save_model, save_scaler
 
dagshub.init(repo_owner=config.DAGSHUB_REPO_OWNER,
             repo_name=config.DAGSHUB_REPO_NAME,
             mlflow=True)

def main():
    """
    Main pipeline orchestrator.

    Workflow:
    1. Setup MLflow experiment tracking
    2. Load and preprocess data
    3. Create model
    4. Train model
    5. Log experiment with MLflow
    6. Save model and scaler artifacts
    """
    print("=" * 80)
    print("DIABETES PREDICTION MODEL TRAINING PIPELINE")
    print("=" * 80)

    try:
        # ====================================================================
        # Step 1: Setup MLflow Experiment
        # ====================================================================
        print("\n[Step 1] Setting up MLflow experiment...")
        setup_mlflow(experiment_name=config.MLFLOW_EXPERIMENT_NAME)

        # ====================================================================
        # Step 2: Load and Preprocess Data
        # ====================================================================
        print("\n[Step 2] Loading and preprocessing data...")
        data_file = os.path.join(config.DATA_PATHS["raw"], "diabetes.csv")

        if not os.path.exists(data_file):
            print(f"ERROR: Data file not found at {data_file}")
            print(
                f"Please place your 'diabetes.csv' file in the {config.DATA_PATHS['raw']} directory."
            )
            sys.exit(1)

        preprocessed_data = preprocess_data(
            filepath=data_file,
            target_column="Outcome",  # Adjust this if your target column has a different name
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )

        X_train = preprocessed_data["X_train"]
        X_test = preprocessed_data["X_test"]
        y_train = preprocessed_data["y_train"]
        y_test = preprocessed_data["y_test"]
        scaler = preprocessed_data["scaler"]

        print(f"  - Training set shape: {X_train.shape}")
        print(f"  - Testing set shape: {X_test.shape}")

        # ====================================================================
        # Step 2.5: Save Processed Data to Disk
        # ====================================================================
        print("\n[Step 2.5] Saving processed data for DVC tracking...")
        from src.data_prep import save_processed_data
        
        save_processed_data(
            X_train, 
            X_test, 
            y_train, 
            y_test, 
            config.DATA_PATHS["processed"]
        )

        # ====================================================================
        # Step 3: Create Model
        # ====================================================================
        print("\n[Step 3] Creating RandomForest model...")
        model = create_model()
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - n_estimators: {model.n_estimators}")
        print(f"  - max_depth: {model.max_depth}")
        print(f"  - random_state: {model.random_state}")

        # ====================================================================
        # Step 4: Train Model
        # ====================================================================
        print("\n[Step 4] Training model...")
        training_results = train_model(model, X_train, y_train, X_test, y_test)

        trained_model = training_results["model"]
        metrics = training_results["metrics"]

        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")

        # ====================================================================
        # Step 5: Log Experiment to MLflow
        # ====================================================================
        print("\n[Step 5] Logging experiment to MLflow...")
        model_params = {
            "n_estimators": config.N_ESTIMATORS,
            "max_depth": config.MAX_DEPTH,
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            

        }

        run_id = log_experiment(trained_model, metrics, model_params, X_test, training_results["y_pred"])
        print(f"  - Run ID: {run_id}")

        # ====================================================================
        # Step 6: Save Model and Scaler Artifacts
        # ====================================================================
        print("\n[Step 6] Saving artifacts...")
        model_path = save_model(trained_model, model_name="diabetes_model.pkl")
        scaler_path = save_scaler(scaler, scaler_name="feature_scaler.pkl")

        print(f"  - Model saved: {model_path}")
        print(f"  - Scaler saved: {scaler_path}")

        # ====================================================================
        # Pipeline Complete
        # ====================================================================
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nArtifacts saved in: {config.MODELS_DIR}")
        print(f"MLflow experiment: {config.MLFLOW_EXPERIMENT_NAME}")
        print(f"Run ID: {run_id}")
        print("\nTo view MLflow UI, run:")
        print("  mlflow ui")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
