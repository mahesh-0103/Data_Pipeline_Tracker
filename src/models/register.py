# src/models/register.py
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def register_best_model(experiment_name: str, model_name: str, metric_name: str, maximize: bool = False):
    """Registers the top run using cross-validated metrics."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
        max_results=1
    )

    if not runs:
        raise ValueError("No valid runs found to register.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    
    # Matches artifact_path="model" from train.py
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.transition_model_version_stage(name=model_name, version=mv.version, stage="Production")
        print(f"[REGISTRY] Successfully promoted {model_name} v{mv.version} to Production.")
        return mv.version, best_run.data.metrics.get(metric_name), best_run_id
    except MlflowException as e:
        print(f"[REGISTRY ERROR] Failed to register: {e}")
        raise