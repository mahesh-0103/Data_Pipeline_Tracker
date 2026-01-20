# workflows/pipeline.py (Updated for Robust Training & Registration)
import sys
from pathlib import Path
import pandas as pd
from prefect import flow, task

# Ensure project root is in path for relative imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data_pipeline.preprocess import preprocess
from src.training.train import train_and_log
from src.models.register import register_best_model

# ---------------------------------------------------------
# Tasks
# ---------------------------------------------------------

@task
def validate_task(df: pd.DataFrame):
    """Basic check to ensure data is not empty before processing."""
    if df.empty:
        raise ValueError("[VALIDATION ERROR] Received an empty DataFrame.")
    print(f"[VALIDATION] Dataframe shape: {df.shape}. Proceeding to preprocess.")
    return df

@task
def preprocess_task(df: pd.DataFrame, target_col: str):
    """Handles scaling, imputation, and target variable cleaning."""
    return preprocess(df, target_col)

@task
def train_task(X_train, y_train, X_val, y_val, experiment_name: str, metric_name: str, mode: str = "auto"):
    """Trains multiple candidates and logs them to MLflow with standardized artifact paths."""
    # Ensure the selection metric is passed down to avoid fallback warnings in train.py
    return train_and_log(
        X_train, y_train, X_val, y_val, 
        experiment_name=experiment_name, 
        metric_name=metric_name,
        mode=mode
    )

@task
def register_task(experiment_name: str, model_name: str, metric_name: str, maximize: bool):
    """Registers the best run found in MLflow under the standardized 'model' path."""
    # Updated to return just the version; other metadata can be fetched from train_result
    return register_best_model(
        experiment_name=experiment_name, 
        model_name=model_name, 
        metric_name=metric_name, 
        maximize=maximize
    )

# ---------------------------------------------------------
# Flows
# ---------------------------------------------------------

@flow(name="MLOps End-to-End Pipeline (File Path)")
def full_pipeline_file_path(
    csv_path: str, 
    target_col: str, 
    metric_name: str = "rmse", 
    maximize: bool = False, 
    model_name: str = "battery_model", 
    experiment_name: str = "mlops_demo", 
    train_mode: str = "auto"
):
    """Flow entry point for CLI usage."""
    from src.data_pipeline.ingest import ingest_and_prepare
    
    @task
    def ingest_task(path: str):
        return ingest_and_prepare(path)
    
    raw_df = ingest_task(csv_path)
    return run_core_pipeline(raw_df, target_col, metric_name, maximize, model_name, experiment_name, train_mode)

@flow(name="MLOps End-to-End Pipeline (Streamlit DF)")
def full_pipeline_streamlit(
    df: pd.DataFrame, 
    target_col: str, 
    metric_name: str = "rmse", 
    maximize: bool = False, 
    model_name: str = "battery_model", 
    experiment_name: str = "mlops_demo", 
    train_mode: str = "auto"
):
    """Flow entry point for Streamlit dashboard usage."""
    return run_core_pipeline(df, target_col, metric_name, maximize, model_name, experiment_name, train_mode)

def run_core_pipeline(df, target_col, metric_name, maximize, model_name, experiment_name, train_mode):
    """Shared core logic for both entry points."""
    print(f"\n===== PIPELINE STARTED FOR {experiment_name.upper()} =====")
    
    # 1. Validation
    df = validate_task(df)

    # 2. Preprocess (Handles the NaN cleaning we added to preprocess.py)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_task(df, target_col)

    # 3. Train (Standardizes artifacts to 'model' folder)
    train_result = train_task(X_train, y_train, X_val, y_val, experiment_name, metric_name, mode=train_mode)
    
    # 4. Register (Now looks specifically for 'model' path)
    version = register_task(experiment_name, model_name, metric_name, maximize)

    print(f"===== PIPELINE COMPLETED: {model_name} v{version} is in Production =====")
    
    return {
        "model_version": version,
        "best_run_id": train_result.get("best_run_id"),
        "best_model_key": train_result.get("best_model_key"),
        "best_metric_value": train_result.get("best_metric_value"),
        "runs": train_result.get("runs", [])
    }

if __name__ == "__main__":
    # Example CLI execution setup
    full_pipeline_file_path(
        csv_path="data/raw/sample.csv", 
        target_col="mileage",
        metric_name="rmse"
    )