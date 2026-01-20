# app_modules/3_train.py
import streamlit as st
import pandas as pd
from src.training.train import train_and_log
from src.data_pipeline.preprocess import preprocess

def _is_ui_regression(df: pd.DataFrame, target_col: str) -> bool:
    """Detects if a target column is likely continuous for UI purposes."""
    if target_col not in df.columns:
        return False
    y = df[target_col].dropna()
    if not pd.api.types.is_numeric_dtype(y):
        return False
    if pd.api.types.is_float_dtype(y):
        return True
    if y.nunique() <= 20:
        return False 
    return True

def app():
    st.subheader("⚙️ Run K-Fold Training Process") 
    st.markdown("---")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Upload dataset first on the **⬆️ Upload Data** page.")
        return

    # Pipeline Configuration
    with st.expander("Pipeline Configuration", expanded=True):
        st.write("**Define parameters for the K-Fold training pipeline.**")
        col_target, col_metric = st.columns(2)
        
        with col_target:
            target_col = st.selectbox("Target Column", list(df.columns), key="train_target_col")
            st.session_state["target_col"] = target_col
            
            is_regression = _is_ui_regression(df, target_col)
            if is_regression:
                default_metric = "rmse"
                metric_options = ["rmse", "mae", "r2", "mse"]
                st.info("Target detected as **Regression**.")
            else:
                default_metric = "f1"
                metric_options = ["f1", "accuracy", "precision", "recall"]
                st.info("Target detected as **Classification**.")

        with col_metric:
            current_metric = st.session_state.get("metric_name", default_metric)
            if current_metric not in metric_options:
                current_metric = default_metric
                
            metric = st.selectbox("Optimization Metric", metric_options, index=metric_options.index(current_metric))
            st.session_state["metric_name"] = metric
            
            maximize = st.checkbox("Higher is better?", value=(metric in ["f1", "accuracy", "r2"]))
            st.session_state["maximize"] = maximize
            
        st.session_state["model_name"] = st.text_input("Registered MLflow Model Name", value="production_model")
        
        st.markdown("---")
        run_prefect = st.checkbox("✅ Use Prefect Orchestration", value=False)

    if st.button("🚀 Run K-Fold Pipeline", type="primary"):
        try:
            if run_prefect:
                # Prefect Flow Execution
                from workflows.pipeline import full_pipeline_streamlit 
                with st.spinner("⏳ Orchestrating via Prefect..."):
                    res = full_pipeline_streamlit(
                        df=df, 
                        target_col=target_col, 
                        metric_name=metric, 
                        maximize=maximize, 
                        model_name=st.session_state["model_name"]
                    )
            else:
                # Direct Execution with NaN Cleaning
                with st.spinner("⏳ Preprocessing & Cleaning Target NaNs..."):
                    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, target_col)
                
                # K-Fold Training
                with st.spinner("🔄 Executing 5-Fold Cross-Validation..."):
                    res = train_and_log(
                        X_train, y_train, X_val, y_val, 
                        experiment_name="mlops_demo", 
                        metric_name=metric, 
                        maximize=maximize
                    )

            # Update Session State with Results
            st.success("✅ Pipeline Finished Successfully.")
            st.session_state["train_runs"] = res.get("runs", [])
            st.session_state["best_model_key"] = res.get("best_model_key")
            st.session_state["best_metric_value"] = res.get("best_metric_value")
            
            # Display best result immediately
            st.balloons()
            st.json(res)

        except Exception as e:
            st.error(f"🚨 Pipeline failed: {e}")
            st.code(traceback.format_exc())

    # Results Summary Table
    runs = st.session_state.get("train_runs")
    if runs:
        st.markdown("---")
        st.subheader("📚 Session Runs (K-Fold Averages)")
        runs_df = pd.DataFrame([{"Model": r.get("model_key"), **r.get("metrics", {})} for r in runs])
        st.dataframe(runs_df, width='stretch')
        
        if st.button("🗑️ Clear Session"):
            st.session_state.pop("train_runs", None)
            st.rerun()