# Diabetes Prediction Model - MLOps Project

A modular, production-ready Python project for training and deploying a Random Forest Classifier for diabetes prediction using MLflow for experiment tracking.

## Project Structure

```
Diabets/
├── config.py                 # Configuration and constants
├── main.py                   # Main pipeline orchestrator
├── requirements.txt          # Python dependencies
├── setup_env.ps1            # Windows virtual environment setup script
├── setup_env.sh             # macOS/Linux virtual environment setup script
├── .gitignore               # Git ignore rules
├── src/
│   ├── data_prep.py         # Data loading and preprocessing
│   ├── model_factory.py     # Model initialization
│   ├── trainer.py           # Training and MLflow logging
│   └── utils.py             # Utility functions for saving/loading artifacts
├── data/
│   ├── raw/                 # Raw data (place diabetes.csv here)
│   └── processed/           # Processed data
└── models/                  # Trained models and scalers
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

#### Windows (PowerShell)

```powershell
# Navigate to project directory
cd path\to\Diabets

# Make sure script execution is enabled (run once)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run the setup script
.\setup_env.ps1

# The virtual environment "Diabet" will be created and activated automatically
```

#### macOS/Linux

```bash
# Navigate to project directory
cd path/to/Diabets

# Make the setup script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh

# The virtual environment "Diabet" will be created and activated automatically
```

### Manual Virtual Environment Setup

If you prefer to set up the virtual environment manually:

#### Windows

```powershell
python -m venv Diabet
.\Diabet\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS/Linux

```bash
python3 -m venv Diabet
source Diabet/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

1. Place your `diabetes.csv` file in the `data/raw/` directory
2. The CSV should have a target column named `Outcome` (adjustable in `main.py` if different)
3. The pipeline will automatically:
   - Load the data
   - Handle missing values (drops rows with NaN)
   - Split into training and testing sets (80/20 by default)
   - Scale features using StandardScaler

### Example CSV Format

```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
...
```

## Configuration

Edit `config.py` to customize:

- `N_ESTIMATORS`: Number of trees in the Random Forest (default: 100)
- `MAX_DEPTH`: Maximum depth of each tree (default: 10)
- `RANDOM_STATE`: Seed for reproducibility (default: 42)
- `TEST_SIZE`: Proportion of data for testing (default: 0.2)
- `DATA_PATHS`: Directory paths for raw and processed data

## Running the Pipeline

**Important:** Always activate the virtual environment before running the pipeline.

### Windows

```powershell
# Activate the environment
.\Diabet\Scripts\Activate.ps1

# Run the pipeline
python main.py
```

### macOS/Linux

```bash
# Activate the environment
source Diabet/bin/activate

# Run the pipeline
python main.py
```

## Pipeline Workflow

The `main.py` orchestrator executes the following steps:

1. **MLflow Setup**: Initializes experiment tracking
2. **Data Loading**: Loads CSV data from `data/raw/`
3. **Preprocessing**: Handles missing values and scales features
4. **Model Creation**: Initializes RandomForestClassifier from config
5. **Training**: Trains the model on training data
6. **Logging**: Logs parameters, metrics, and model to MLflow
7. **Artifacts**: Saves model and scaler to `models/`

### Pipeline Output

```
======================== DIABETES PREDICTION MODEL TRAINING PIPELINE ========================
[Step 1] Setting up MLflow experiment...
[Step 2] Loading and preprocessing data...
[Step 3] Creating RandomForest model...
[Step 4] Training model...
[Step 5] Logging experiment to MLflow...
[Step 6] Saving artifacts...

Metrics:
  - Accuracy:  0.8234
  - F1-Score:  0.8156
  - Precision: 0.8342
```

## MLflow Experiment Tracking

### View Training Experiments

Start the MLflow UI:

```bash
mlflow ui
```

Then open your browser and navigate to `http://127.0.0.1:5000`

### Key Features

- Compare model runs
- View logged parameters and metrics
- Access trained model artifacts
- Track experiment history

## Module Documentation

### config.py

Central configuration module. Contains all constants and paths.

- Define hyperparameters
- Set data paths
- Configure MLflow tracking

### src/data_prep.py

Data loading and preprocessing functions:

- `load_data()`: Load CSV files
- `handle_missing_values()`: Handle NaN values (drop, mean, median, ffill, bfill)
- `scale_features()`: Apply StandardScaler to features
- `preprocess_data()`: Complete preprocessing pipeline

### src/model_factory.py

Model initialization:

- `create_model()`: Initialize RandomForestClassifier with config parameters

### src/trainer.py

Training and MLflow logging:

- `train_model()`: Train model and calculate metrics
- `log_experiment()`: Log to MLflow with parameters and metrics
- `setup_mlflow()`: Initialize MLflow experiment

### src/utils.py

Artifact persistence utilities:

- `save_model()`: Save trained model using joblib
- `load_model()`: Load trained model from disk
- `save_scaler()`: Save fitted scaler using joblib
- `load_scaler()`: Load scaler from disk
- `save_artifact()` / `load_artifact()`: Generic utility functions

### main.py

Main orchestrator that integrates all components and executes the pipeline.

## Artifacts

After running the pipeline, you'll find:

- **models/diabetes_model.pkl**: Trained Random Forest model
- **models/feature_scaler.pkl**: Fitted StandardScaler for feature normalization
- **mlruns/**: MLflow experiment tracking directory

## Using Trained Model for Predictions

```python
from src.utils import load_model, load_scaler
import numpy as np

# Load the model and scaler
model = load_model("diabetes_model.pkl")
scaler = load_scaler("feature_scaler.pkl")

# Prepare new data (example: 8 features)
new_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Scale features
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

## Optional: DagShub Integration

To track experiments on DagShub (requires credentials):

1. Uncomment lines in `main.py`:

```python
import dagshub
dagshub.init(repo_owner="your-username", repo_name="diabetes-prediction", mlflow=True)
```

2. Set your DagShub credentials

## Code Quality Standards

This project follows best practices:

- ✅ Modular architecture with clear separation of concerns
- ✅ Comprehensive docstrings for all functions
- ✅ Cross-platform file path handling (Windows/macOS/Linux)
- ✅ Proper error handling and validation
- ✅ Use of `if __name__ == "__main__"` blocks
- ✅ Production-ready MLflow integration
- ✅ Clean, readable code with meaningful variable names

## Requirements

All dependencies are listed in `requirements.txt`:

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library
- **mlflow**: Experiment tracking and model registry
- **dagshub**: Data science platform integration
- **joblib**: Serialization of Python objects
- **numpy**: Numerical computing

## Troubleshooting

### "Python not found" error

- Ensure Python 3.8+ is installed
- Add Python to system PATH

### Virtual environment activation fails

- Windows: Enable script execution with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- macOS/Linux: Run `chmod +x setup_env.sh`

### "diabetes.csv not found" error

- Place your CSV file in `data/raw/` directory
- Ensure file name is exactly `diabetes.csv`

### MLflow UI not accessible

- Ensure MLflow is installed: `pip install mlflow`
- Run `mlflow ui` from the project directory
- Open `http://127.0.0.1:5000` in your browser

## Next Steps

1. Add your diabetes dataset to `data/raw/diabetes.csv`
2. Run the pipeline: `python main.py`
3. View results in MLflow UI
4. Compare model runs and track metrics
5. Deploy the model for inference

## License

This is an educational project for demonstration purposes.

## Support

For issues or questions:

1. Check the docstrings in each module
2. Review the MLflow experiment history
3. Verify data format matches expectations
4. Check that all dependencies are installed correctly

---

**Happy training! 🚀**
#   D i a b e t  
 