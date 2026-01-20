# src/training/train.py
import math
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# core models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

# optional model libs (import safely)
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:
    CatBoostRegressor = CatBoostClassifier = None

# --------------------------
# Helpers: detection
# --------------------------
def _is_regression_target(y: pd.Series) -> bool:
    if y.dtype == 'object' or y.dtype.name == 'category' or pd.api.types.is_string_dtype(y):
        return False
    if pd.api.types.is_float_dtype(y):
        return True
    if y.nunique() <= 20:
        return False 
    return True

def _get_regression_models():
    models = {
        "LinearReg": LinearRegression(), "RidgeReg": Ridge(), "LassoReg": Lasso(),
        "KNNReg": KNeighborsRegressor(), "SVR": SVR(),
        "RandomForestReg": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), 
        "ExtraTreesReg": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoostReg": GradientBoostingRegressor(random_state=42),
        "HistGradientBoostReg": HistGradientBoostingRegressor(random_state=42),
        "AdaBoostReg": AdaBoostRegressor(random_state=42),
        "MLPReg": MLPRegressor(max_iter=500, random_state=42),
    }
    if xgb is not None: models["XGBReg"] = xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if lgb is not None: models["LGBMReg"] = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if CatBoostRegressor is not None: models["CatBoostReg"] = CatBoostRegressor(verbose=0, random_state=42)
    return models

def _get_classification_models():
    models = {
        "LogisticReg": LogisticRegression(max_iter=500), "KNNClf": KNeighborsClassifier(), "SVC": SVC(probability=False),
        "RandomForestClf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "ExtraTreesClf": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoostClf": GradientBoostingClassifier(random_state=42),
        "HistGradientBoostClf": HistGradientBoostingClassifier(random_state=42),
        "AdaBoostClf": AdaBoostClassifier(random_state=42),
        "MLPClf": MLPClassifier(max_iter=500, random_state=42),
    }
    if xgb is not None: models["XGBClf"] = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    if lgb is not None: models["LGBMClf"] = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if CatBoostClassifier is not None: models["CatBoostClf"] = CatBoostClassifier(verbose=0, random_state=42)
    return models

# --------------------------
# Main training + logging function
# --------------------------
def train_and_log(X_train, y_train, X_val, y_val, experiment_name: str = "mlops_demo", candidate_models: list | None = None, metric_name: str | None = None, maximize: bool | None = None, mode: str = "auto"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # Combine subsets for K-Fold to maximize data utility
    X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)

    is_reg = _is_regression_target(y_full) if mode == "auto" else (mode == "regression")
    
    if not is_reg and not pd.api.types.is_integer_dtype(y_full):
        le = LabelEncoder()
        y_full = pd.Series(le.fit_transform(y_full.astype(str)), name=y_full.name)

    models_to_try = _get_regression_models() if is_reg else _get_classification_models()
    if candidate_models:
        models_to_try = {k: v for k, v in models_to_try.items() if k in candidate_models}

    selection_metric = metric_name or ("rmse" if is_reg else "f1")
    if maximize is None: 
        maximize = True if selection_metric in ("accuracy", "precision", "recall", "f1", "r2") else False

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scorers = {'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'} if is_reg else \
              {'accuracy': 'accuracy', 'f1': 'f1_weighted', 'precision': 'precision_weighted', 'recall': 'recall_weighted'}

    runs = []
    best_metric, best_run_id, best_model_key = None, None, None

    for key, model in models_to_try.items():
        with mlflow.start_run(run_name=key):
            try: 
                # PERFORM CROSS VALIDATION
                cv_results = cross_validate(model, X_full, y_full, cv=kf, scoring=scorers)
                
                if is_reg:
                    metrics = {
                        "mse": float(np.abs(np.mean(cv_results['test_mse']))),
                        "mae": float(np.abs(np.mean(cv_results['test_mae']))),
                        "r2": float(np.mean(cv_results['test_r2'])),
                        "rmse": float(np.sqrt(np.abs(np.mean(cv_results['test_mse']))))
                    }
                else:
                    metrics = {m: float(np.mean(cv_results[f'test_{m}'])) for m in scorers.keys()}

                for m_k, m_v in metrics.items():
                    mlflow.log_metric(m_k, -1.0 if np.isnan(m_v) else m_v)

                # Final Fit and Standardized Path
                model.fit(X_full, y_full)
                mlflow.log_param("model_key", key)
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=X_full.head(1))

                run_id = mlflow.active_run().info.run_id
                runs.append({"model_key": key, "run_id": run_id, "metrics": metrics})

                cur_val = metrics.get(selection_metric)
                if best_metric is None or (maximize and cur_val > best_metric) or (not maximize and cur_val < best_metric):
                    best_metric, best_run_id, best_model_key = cur_val, run_id, key

            except Exception as e:
                print(f"[TRAIN ERROR] Skipping {key}: {e}")
                continue

    return {"is_regression": is_reg, "runs": runs, "best_model_key": best_model_key, "best_run_id": best_run_id, "best_metric_name": selection_metric, "best_metric_value": best_metric}