"""
Data loading utilities for clinical deterioration prediction datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_3year_dataset(data_path: str = "data/CNUH_3Y.csv") -> pd.DataFrame:
    """
    Load the 3-year dataset (CNUH_3Y.csv).
    
    Parameters:
    -----------
    data_path : str
        Path to the 3-year dataset CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset with proper data types
    """
    df = pd.read_csv(data_path)
    
    # Convert time columns to datetime
    time_cols = ['measurement_time', 'event_time', 'detection_time']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ensure target is numeric
    if 'target' in df.columns:
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
    
    return df


def load_10year_dataset(data_path: str = "data/10yrs_proc.csv") -> pd.DataFrame:
    """
    Load the 10-year dataset (10yrs_proc.csv).
    
    Parameters:
    -----------
    data_path : str
        Path to the 10-year dataset CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset with proper data types
    """
    df = pd.read_csv(data_path)
    
    # Remove the first column if it's an index column
    if df.columns[0] == 'Unnamed: 0':
        df = df.drop(columns=['Unnamed: 0'])
    
    # Convert time columns to datetime
    time_cols = ['adjusted_time', 'measurement_time', 'event_time', 'detection_time']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ensure target is numeric
    if 'target' in df.columns:
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
    
    return df


def validate_dataset(df: pd.DataFrame, dataset_name: str = "dataset") -> dict:
    """
    Validate dataset structure and return summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    dataset_name : str
        Name of the dataset for reporting
        
    Returns:
    --------
    dict
        Validation summary with statistics
    """
    validation = {
        'name': dataset_name,
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Check for target column
    if 'target' in df.columns:
        validation['target_distribution'] = df['target'].value_counts().to_dict()
        validation['target_missing'] = df['target'].isnull().sum()
    
    return validation


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Categorize features into groups (vital signs, lab values, demographics, etc.).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    dict
        Dictionary with feature groups
    """
    feature_groups = {
        'vital_signs': [],
        'lab_values': [],
        'demographics': [],
        'temporal': [],
        'other': []
    }
    
    # Define feature categories
    vital_signs_keywords = ['HR', 'RR', 'SBP', 'SaO2', 'BT', 'TS']
    lab_keywords = ['WBC', 'Hgb', 'platelet', 'ALT', 'AST', 'Albumin', 
                    'Alkaline', 'BUN', 'CRP', 'Chloride', 'Creatinin', 
                    'Glucose', 'Lactate', 'Potassium', 'Sodium', 'bilirubin', 
                    'calcium', 'protein']
    demo_keywords = ['Age', 'Gender']
    temporal_keywords = ['time', 'TS', 'is_abn']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(kw.lower() in col_lower for kw in vital_signs_keywords):
            feature_groups['vital_signs'].append(col)
        elif any(kw.lower() in col_lower for kw in lab_keywords):
            feature_groups['lab_values'].append(col)
        elif any(kw.lower() in col_lower for kw in demo_keywords):
            feature_groups['demographics'].append(col)
        elif any(kw.lower() in col_lower for kw in temporal_keywords):
            feature_groups['temporal'].append(col)
        elif col not in ['Patient', 'target']:
            feature_groups['other'].append(col)
    
    return feature_groups
