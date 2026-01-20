# streamlit_app/main.py
import streamlit as st
from pathlib import Path
import sys
import importlib.util
import traceback
import os

# --- ENVIRONMENT & PATH SETUP ---
# Silence GitPython noise for a cleaner local UI experience
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Add project root to sys.path so 'from src...' imports work correctly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- SET PAGE CONFIG ---
st.set_page_config(
    page_title="MLOps Pipeline Tracker | K-Fold Edition",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "# MLOps Pipeline Tracker\nDeveloping robust ML systems with K-Fold Validation."}
)

# --- SESSION DEFAULTS ---
# Updated to handle K-Fold metadata
defaults = {
    "df": None,
    "train_runs": None,
    "target_col": None,
    "metric_name": "rmse",
    "maximize": False,
    "model_name": "production_model",
    "best_model_key": None,
    "best_metric_value": None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- NAVIGATION SETUP ---
PAGES_DIR = Path(__file__).parent / "app_modules"
NAVIGATION_ORDER = {
    "1_upload": "⬆️ Upload Data",
    "2_eda": "📊 Data Analysis",
    "3_train": "⚙️ Run K-Fold Process",
    "4_model_compare": "🏆 Compare Averages",
    "5_visualize": "📈 Performance Reports"
}

def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Data Pipeline Tracker")
    st.markdown("**Organized K-Fold Workflow**")
    st.markdown("---")
    
    st.caption("Workflow Stages")
    selected_label = st.radio(
        "Select Stage", 
        list(NAVIGATION_ORDER.values()), 
        index=0, 
        key="sidebar_nav",
        label_visibility="collapsed"
    )
    
    # --- NEW: PRODUCTION STATUS TRACKER ---
    st.markdown("---")
    with st.expander("🚀 Production Status", expanded=True):
        if st.session_state.best_model_key:
            st.success(f"**Model:** {st.session_state.best_model_key}")
            st.info(f"**Avg {st.session_state.metric_name.upper()}:** {st.session_state.best_metric_value:.4f}")
        else:
            st.warning("No model registered yet.")

# --- MAIN CONTENT AREA ---
MAIN_HEADING_MAP = {
    "⬆️ Upload Data": "Data Ingestion & Cleaning",
    "📊 Data Analysis": "Exploratory Analysis (EDA)",
    "⚙️ Run K-Fold Process": "Training: 5-Fold Cross-Validation",
    "🏆 Compare Averages": "Results: Statistical Review",
    "📈 Performance Reports": "Visualization & Reporting"
}

st.markdown(f"# {MAIN_HEADING_MAP.get(selected_label, 'MLOps Dashboard')}")
st.write("---")

# Load and run the selected module
file_stem = [k for k, v in NAVIGATION_ORDER.items() if v == selected_label][0]
page_path = PAGES_DIR / f"{file_stem}.py"

try:
    module = import_module_from_path(f"module_{file_stem}", page_path)
    module.app()
except Exception as e:
    st.error(f"❌ Error loading {selected_label}")
    st.exception(e)
    st.code(traceback.format_exc())