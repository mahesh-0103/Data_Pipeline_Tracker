   # 📚 Data Pipeline Tracker: Professional MLOps Dashboard
    
    **Data Pipeline Tracker** is an end-to-end Machine Learning Operations (MLOps) platform designed to automate the lifecycle of predictive models. Built with a focus on statistical rigor and production-ready architecture, it enables seamless transitions from raw data ingestion to registered production candidates.
    
    ![MLOps Lifecycle](https://raw.githubusercontent.com/mahesh-0103/Data_Pipeline_Tracker/main/architectural_diagram.png)
    
    ## 🚀 Advanced Features
    
    * **Statistical Validation (K-Fold):** Implements **5-Fold Cross-Validation** to ensure performance metrics (RMSE, R², F1-Score) represent generalized model performance rather than single-split bias.
    * **Intelligent Preprocessing:** Features automated **NaN cleaning for targets**, **Label Encoding** for categorical safety, and **Datetime Feature Extraction** to transform timestamp strings into numeric signals.
    * **Orchestration with Prefect:** Uses a task-based orchestration layer to manage the sequence of Ingestion, Validation, Preprocessing, Training, and Registration.
    * **Experiment Tracking via MLflow:** Integrated with an **MLflow SQLite backend** for robust local experiment management, logging hyperparameters, artifacts, and cross-validated metrics.
    * **Production Registry:** Automated model versioning that transitions the statistically "best" run to the **"Production"** stage within the MLflow Model Registry.
    
    ## ⚙️ Technical Stack
    
    | Category | Component | Purpose |
    | :--- | :--- | :--- |
    | **Interface** | Streamlit | Interactive multi-stage MLOps dashboard. |
    | **Orchestration** | Prefect | Automates task sequencing and error handling. |
    | **Experiment Tracking**| MLflow (SQLite) | Persistent logging of metrics and model artifacts. |
    | **Model Registry** | MLflow Registry | Standardized version control and deployment staging. |
    | **ML Libraries** | Scikit-learn, XGBoost | Implementation of 10+ regression and classification models. |
    | **Data Engineering** | Pandas, NumPy | Cleaning, feature extraction, and K-Fold pooling. |
    
    ## 🧭 Workflow Architecture
    
    The dashboard is organized into five modular stages to mimic a professional data science workflow:
    
    1. **⬆️ Data Ingestion:** Supports CSV, XLSX, PKL, and JSON with automatic type casting to prevent Arrow errors.
    2. **📊 Analysis (EDA):** Generates univariate distributions, categorical pie charts, and correlation heatmaps to identify feature relationships.
    3. **⚙️ Training Pipeline:** Configures the **K-Fold Cross-Validation** loop, allowing users to select optimization metrics like RMSE or F1.
    4. **🏆 Comparative Review:** Visualizes mean CV scores across all candidates to ensure a reliable production selection.
    5. **📈 Performance Reports:** Generates full visual reports including histograms and pairwise scatter plots for the entire dataset.
    
    ---
    
    ## 🛠️ Local Installation & Reset
    
    ### 1. Clone & Environment Setup
    ```bash
    git clone [https://github.com/mahesh-0103/Data_Pipeline_Tracker.git](https://github.com/mahesh-0103/Data_Pipeline_Tracker.git)
    cd Data_Pipeline_Tracker
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate  # macOS/Linux
    pip install -r requirements.txt
    

### 2\. Initialize Backend Servers

To ensure a clean initialization of the tracking metadata and resolve potential synchronization errors:

**Terminal A: MLflow (Tracking Server)**

Bash

    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
    

**Terminal B: Prefect (Orchestration Server)**

Bash

    prefect server database reset -y
    prefect server start
    

### 3\. Launch Frontend

**Terminal C: Streamlit Dashboard**

Bash

    streamlit run streamlit_app/main.py
    

* * *


