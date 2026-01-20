# src/data_pipeline/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def preprocess(df: pd.DataFrame, target_col: str, max_sample_size: int = 20000):
    """
    Cleans target NaNs, handles datetime conversion, feature imputation/scaling.
    """
    # 1. CLEAN TARGET: Drop NaN in Target
    if df[target_col].isnull().any():
        initial_count = len(df)
        df = df.dropna(subset=[target_col])
        print(f"[PREPROCESS] Dropped {initial_count - len(df)} rows with NaN in target.")

    # 2. DATETIME HANDLING: Convert timestamps to numeric features
    for col in df.columns:
        if col == target_col:
            continue
            
        # Check if column is object/string and looks like a date
        if df[col].dtype == 'object':
            try:
                # Attempt to convert to datetime
                temp_date = pd.to_datetime(df[col])
                print(f"[PREPROCESS] Converting datetime column: {col}")
                
                # Extract numeric features from date
                df[f"{col}_hour"] = temp_date.dt.hour
                df[f"{col}_day"] = temp_date.dt.day
                df[f"{col}_month"] = temp_date.dt.month
                df[f"{col}_year"] = temp_date.dt.year
                
                # Drop the original string date column
                df = df.drop(columns=[col])
            except (ValueError, TypeError):
                # Not a date column, leave it for LabelEncoding or dropping
                pass

    # 3. SAMPLING
    if len(df) > max_sample_size:
        df = df.sample(n=max_sample_size, random_state=42).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. MODE DETECTION
    is_regression = pd.api.types.is_float_dtype(y) or (pd.api.types.is_numeric_dtype(y) and y.nunique() > 20)
    
    # 5. CATEGORICAL ENCODING: Handle remaining string columns
    remaining_cats = X.select_dtypes(include=['object']).columns
    for col in remaining_cats:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    if not is_regression and not pd.api.types.is_integer_dtype(y):
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y.astype(str)), name=y.name)

    # 6. SPLIT FOR K-FOLD POOL
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=42)
    
    # 7. NUMERIC PROCESSING (Imputation/Scaling)
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy="median") 
        scaler = StandardScaler()
        
        X_train[numeric_cols] = scaler.fit_transform(imputer.fit_transform(X_train[numeric_cols]))
        X_val[numeric_cols] = scaler.transform(imputer.transform(X_val[numeric_cols]))
        X_test[numeric_cols] = scaler.transform(imputer.transform(X_test[numeric_cols]))

    print(f"[PREPROCESS] Data processed. Features: {X.shape[1]}")
    return (X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True), 
            y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True))