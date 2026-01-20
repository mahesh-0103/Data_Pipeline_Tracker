# main.py
import argparse
import logging
import os
import sys
from pathlib import Path

# --- ENVIRONMENT SETUP ---
# Silence GitPython executable warnings
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Ensure imports work from project root for relative paths
sys.path.append(str(Path(__file__).resolve().parents[0]))

# Import the robust full pipeline
from workflows.pipeline import full_pipeline_file_path

# Configure logging to track pipeline progress
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mlops_main")

def run_cli(file_path: str, target_col: str, metric_name: str, maximize: bool, model_name: str):
    """
    Executes the robust K-Fold MLOps pipeline on any provided dataset.
    """
    p = Path(file_path)
    if not p.exists():
        logger.error(f"❌ File not found: {file_path}")
        sys.exit(1)

    logger.info(f"🚀 Initializing Robust Pipeline: {p.name}")
    logger.info(f"Target: {target_col} | Optimized Metric: {metric_name} (Maximize={maximize})")
    
    try:
        # Executes the file-based flow which now includes:
        # 1. Automatic Target NaN cleaning
        # 2. 5-Fold Cross-Validation for reliable metrics
        # 3. Standardized artifact registration
        result = full_pipeline_file_path(
            csv_path=str(p.resolve()), 
            target_col=target_col, 
            metric_name=metric_name,
            maximize=maximize, 
            model_name=model_name
        )
        
        logger.info(f"✅ Pipeline Completed Successfully!")
        logger.info(f"Best Model Key: {result.get('best_model_key')}")
        logger.info(f"Registered Version: {result.get('model_version')}")
        return result

    except Exception as e:
        logger.error(f"❌ Pipeline failed during execution: {e}")
        sys.exit(1)

def create_arg_parser():
    """Defines command-line arguments for dynamic file processing."""
    p = argparse.ArgumentParser(description="MLOps Universal K-Fold Pipeline")
    
    # Required positional argument for the dataset path
    p.add_argument("file", type=str, help="Path to dataset (CSV, XLSX, Parquet, JSON, etc.)")
    
    # Target column must be specified
    p.add_argument("--target", "-t", type=str, required=True, help="Column name to predict")
    
    # Metric selection with fallback to rmse/f1
    p.add_argument("--metric", "-m", type=str, default="rmse", 
                   help="Metric for model selection (rmse, mae, r2, f1, accuracy)")
    
    # Flag to determine if higher values are better
    p.add_argument("--maximize", action="store_true", 
                   help="Set if higher metric value is better (e.g., accuracy, f1, r2)")
    
    # Registry name for the production model
    p.add_argument("--model-name", type=str, default="production_model", 
                   help="Name of the model in MLflow Registry")
    
    return p

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Initiate the CLI run logic
    run_cli(
        file_path=args.file, 
        target_col=args.target, 
        metric_name=args.metric,
        maximize=args.maximize, 
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()