"""
Data preprocessing utilities for time-to-event models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from scipy import stats


def handle_missing_values(df: pd.DataFrame,
                         feature_cols: List[str],
                         strategy: str = 'median',
                         categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values in feature columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    feature_cols : List[str]
        List of feature column names
    strategy : str
        Imputation strategy: 'median', 'mean', 'mode', 'forward_fill', 'drop'
    categorical_cols : List[str], optional
        List of categorical column names (use mode for these)
        
    Returns:
    --------
    pd.DataFrame
        Dataset with imputed missing values
    """
    df = df.copy()
    
    if categorical_cols is None:
        categorical_cols = []
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
        
        if col in categorical_cols or df[col].dtype == 'object':
            # Use mode for categorical
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col].fillna(mode_value[0], inplace=True)
            else:
                df[col].fillna(0, inplace=True)
        else:
            # Numeric columns
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'mode':
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
            elif strategy == 'forward_fill':
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(df[col].median(), inplace=True)  # Fill remaining with median
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
    
    return df


def detect_outliers_iqr(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    feature_cols : List[str]
        List of feature column names
        
    Returns:
    --------
    dict
        Dictionary with outlier information for each feature
    """
    outliers = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        data = df[col].dropna()
        if len(data) == 0:
            continue
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_pct = outlier_count / len(data) * 100
        
        outliers[col] = {
            'count': int(outlier_count),
            'percentage': float(outlier_pct),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    
    return outliers


def handle_outliers(df: pd.DataFrame,
                   feature_cols: List[str],
                   method: str = 'clip',
                   outlier_info: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """
    Handle outliers in feature columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    feature_cols : List[str]
        List of feature column names
    method : str
        Method: 'clip' (clip to bounds), 'remove' (remove rows), 'winsorize'
    outlier_info : dict, optional
        Pre-computed outlier information from detect_outliers_iqr
        
    Returns:
    --------
    pd.DataFrame
        Dataset with outliers handled
    """
    df = df.copy()
    
    if outlier_info is None:
        outlier_info = detect_outliers_iqr(df, feature_cols)
    
    for col in feature_cols:
        if col not in df.columns or col not in outlier_info:
            continue
        
        bounds = outlier_info[col]
        lower_bound = bounds['lower_bound']
        upper_bound = bounds['upper_bound']
        
        if method == 'clip':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'winsorize':
            # Winsorize to 5th and 95th percentiles
            lower_pct = df[col].quantile(0.05)
            upper_pct = df[col].quantile(0.95)
            df[col] = df[col].clip(lower=lower_pct, upper=upper_pct)
        elif method == 'remove':
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]
    
    return df


def normalize_features(df: pd.DataFrame,
                      feature_cols: List[str],
                      method: str = 'standardize',
                      scaler_params: Optional[Dict[str, Any]] = None) -> tuple:
    """
    Normalize/standardize features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    feature_cols : List[str]
        List of feature column names
    method : str
        Normalization method: 'standardize', 'minmax', 'robust'
    scaler_params : dict, optional
        Parameters for scaler (mean, std, min, max, etc.)
        
    Returns:
    --------
    tuple
        (normalized_df, scaler_info)
    """
    df = df.copy()
    
    if scaler_params is None:
        scaler_params = {}
    
    scaler_info = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Skip non-numeric columns (datetime, object, etc.)
        if df[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            continue
        
        data = df[col].dropna()
        if len(data) == 0:
            continue
        
        if method == 'standardize':
            mean_val = data.mean() if 'mean' not in scaler_params.get(col, {}) else scaler_params[col]['mean']
            std_val = data.std() if 'std' not in scaler_params.get(col, {}) else scaler_params[col]['std']
            if std_val == 0:
                std_val = 1
            df[col] = (df[col] - mean_val) / std_val
            scaler_info[col] = {'mean': float(mean_val), 'std': float(std_val), 'method': 'standardize'}
        
        elif method == 'minmax':
            min_val = data.min() if 'min' not in scaler_params.get(col, {}) else scaler_params[col]['min']
            max_val = data.max() if 'max' not in scaler_params.get(col, {}) else scaler_params[col]['max']
            if max_val == min_val:
                df[col] = 0
            else:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            scaler_info[col] = {'min': float(min_val), 'max': float(max_val), 'method': 'minmax'}
        
        elif method == 'robust':
            median_val = data.median() if 'median' not in scaler_params.get(col, {}) else scaler_params[col]['median']
            iqr_val = data.quantile(0.75) - data.quantile(0.25)
            if iqr_val == 0:
                iqr_val = 1
            df[col] = (df[col] - median_val) / iqr_val
            scaler_info[col] = {'median': float(median_val), 'iqr': float(iqr_val), 'method': 'robust'}
    
    return df, scaler_info


def create_temporal_features(df: pd.DataFrame,
                            patient_col: str = 'Patient',
                            measurement_time_col: str = 'measurement_time',
                            admission_time_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create temporal features (time since admission, time of day, etc.).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    patient_col : str
        Column name for patient ID
    measurement_time_col : str
        Column name for measurement time
    admission_time_col : str, optional
        Column name for admission time. If None, uses first measurement per patient.
        
    Returns:
    --------
    pd.DataFrame
        Dataset with temporal features added
    """
    df = df.copy()
    
    df[measurement_time_col] = pd.to_datetime(df[measurement_time_col], errors='coerce')
    
    # Time since admission (hours)
    if admission_time_col is None:
        # Use first measurement time per patient as admission time
        admission_times = df.groupby(patient_col)[measurement_time_col].min()
        df['admission_time'] = df[patient_col].map(admission_times)
    else:
        df['admission_time'] = pd.to_datetime(df[admission_time_col], errors='coerce')
    
    df['time_since_admission_hours'] = (
        (df[measurement_time_col] - df['admission_time']).dt.total_seconds() / 3600
    )
    df['time_since_admission_hours'] = df['time_since_admission_hours'].fillna(0)
    
    # Time of day features
    df['hour_of_day'] = df[measurement_time_col].dt.hour
    df['day_of_week'] = df[measurement_time_col].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df
