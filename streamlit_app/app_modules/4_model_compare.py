# app_modules/4_model_compare.py
import streamlit as st
import pandas as pd
import json

# --- MAPPING FOR VERBOSE DISPLAY ---
# Converts technical keys to clear, professional names
VERBOSE_MODEL_MAP = {
    "LinearReg": "Linear Regression", "RidgeReg": "Ridge Regression", "LassoReg": "Lasso Regression",
    "KNNReg": "KNN Regressor", "SVR": "Support Vector Reg.", "RandomForestReg": "Random Forest Regressor", 
    "ExtraTreesReg": "Extra Trees Regressor", "GradientBoostReg": "Gradient Boost Reg.",
    "HistGradientBoostReg": "Hist Gradient Boost Reg.", "AdaBoostReg": "AdaBoost Regressor",
    "MLPReg": "MLP Regressor", "XGBReg": "XGBoost Regressor", "LGBMReg": "LGBM Regressor", 
    "CatBoostReg": "CatBoost Regressor",
    
    "LogisticReg": "Logistic Regression", "KNNClf": "KNN Classifier", "SVC": "Support Vector Clf.",
    "RandomForestClf": "Random Forest Classifier", "ExtraTreesClf": "Extra Trees Classifier",
    "GradientBoostClf": "Gradient Boost Clf.", "HistGradientBoostClf": "Hist Gradient Boost Clf.",
    "AdaBoostClf": "AdaBoost Classifier", "MLPClf": "MLP Classifier", "XGBClf": "XGBoost Classifier", 
    "LGBMClf": "LGBM Classifier", "CatBoostClf": "CatBoost Classifier",
}

def get_display_name(key: str) -> str:
    """Helper for user-friendly name display."""
    if not key:
        return "N/A"
    return VERBOSE_MODEL_MAP.get(key, key.replace('_', ' ').title())

def app():
    st.subheader("🏆 Comparative Results (K-Fold Averages)")
    st.markdown("---")
    
    runs = st.session_state.get("train_runs")
    if not runs:
        st.info("No training runs available. Please execute the pipeline on the **⚙️ Run Process** page.")
        return

    st.subheader("All Training Runs")
    st.caption("Metrics shown are the mean averages calculated via 5-Fold Cross-Validation.")
    
    # --- Dynamic Metric Display ---
    # Extract keys from the metrics dictionary which now contains K-Fold means
    runs_data = [r.get("metrics", {}) for r in runs]
    all_metrics = sorted(list(set().union(*(d.keys() for d in runs_data))))
    
    # Build dataframe for display
    comparison_list = []
    for r in runs:
        row = {"Model": get_display_name(r.get("model_key"))}
        row.update(r.get("metrics", {}))
        comparison_list.append(row)
    
    df = pd.DataFrame(comparison_list)
    
    # Highlight the best values (Min for errors, Max for R2/F1)
    style_df = df.style.format(precision=4)
    if 'r2' in df.columns:
        style_df = style_df.highlight_max(subset=['r2'], color='#2E7D32')
    if 'f1' in df.columns:
        style_df = style_df.highlight_max(subset=['f1'], color='#2E7D32')
    if 'rmse' in df.columns:
        style_df = style_df.highlight_min(subset=['rmse'], color='#2E7D32')

    st.dataframe(style_df, use_container_width=True)

    st.markdown("---")

    # --- Best Model Summary Card ---
    st.subheader("Selected Production Candidate")
    
    best_metric = st.session_state.get("best_metric_name", "N/A")
    best_value = st.session_state.get("best_metric_value", None)
    best_model_key = st.session_state.get("best_model_key", None)

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Optimization Metric", best_metric.upper())
    
    display_val = f"{best_value:.4f}" if isinstance(best_value, (int, float)) else "—"
    col_2.metric("Mean CV Score", display_val)
    
    col_3.metric("Best Architecture", get_display_name(best_model_key))

    st.markdown("---")

    st.subheader("Export Results")
    st.download_button(
        label="⬇️ Download Runs Metadata (JSON)", 
        data=json.dumps(runs, indent=4, default=str), 
        file_name="kfold_runs_metadata.json",
        mime="application/json",
        help="Download full details of the cross-validation results."
    )