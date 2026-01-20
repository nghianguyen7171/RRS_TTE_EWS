"""
Feature engineering utilities for time-to-event models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize features into groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
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
    temporal_keywords = ['time', 'hour', 'day', 'weekend', 'sin', 'cos']
    
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
        elif col not in ['Patient', 'target', 'time_to_event', 'event_occurred',
                        'measurement_time', 'event_time', 'detection_time', 'is_abn']:
            feature_groups['other'].append(col)
    
    return feature_groups


def create_interaction_features(df: pd.DataFrame,
                               feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Create interaction features (e.g., vital signs × lab values).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    feature_groups : dict, optional
        Feature groups dictionary. If None, auto-detect.
        
    Returns:
    --------
    pd.DataFrame
        Dataset with interaction features added
    """
    df = df.copy()
    
    if feature_groups is None:
        feature_groups = get_feature_groups(df)
    
    # Age × Lab values interactions
    if 'Age' in df.columns:
        age_col = 'Age'
        for lab_col in feature_groups.get('lab_values', [])[:5]:  # Limit to avoid explosion
            if lab_col in df.columns:
                df[f'{age_col}_x_{lab_col}'] = df[age_col] * df[lab_col]
    
    # Vital signs × Lab values interactions (key ones)
    key_vitals = feature_groups.get('vital_signs', [])[:3]
    key_labs = feature_groups.get('lab_values', [])[:3]
    
    for vital in key_vitals:
        if vital not in df.columns:
            continue
        for lab in key_labs:
            if lab not in df.columns:
                continue
            df[f'{vital}_x_{lab}'] = df[vital] * df[lab]
    
    return df


def create_aggregated_features(df: pd.DataFrame,
                              patient_col: str = 'Patient',
                              feature_cols: Optional[List[str]] = None,
                              window: str = 'all') -> pd.DataFrame:
    """
    Create aggregated features from patient history.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    patient_col : str
        Column name for patient ID
    feature_cols : list, optional
        List of feature columns to aggregate
    window : str
        Aggregation window: 'all' (all previous), '24h' (last 24 hours)
        
    Returns:
    --------
    pd.DataFrame
        Dataset with aggregated features added
    """
    df = df.copy()
    
    if feature_cols is None:
        feature_groups = get_feature_groups(df)
        feature_cols = (feature_groups.get('vital_signs', []) + 
                       feature_groups.get('lab_values', [])[:5])
    
    # Sort by patient and time
    df = df.sort_values([patient_col, 'measurement_time'])
    
    # Group by patient
    grouped = df.groupby(patient_col)
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Rolling statistics
        df[f'{col}_mean_prev'] = grouped[col].transform(lambda x: x.expanding().mean().shift(1))
        df[f'{col}_std_prev'] = grouped[col].transform(lambda x: x.expanding().std().shift(1))
        df[f'{col}_trend'] = grouped[col].transform(lambda x: x.diff())
        
        # Fill NaN with 0 for first measurement per patient
        df[f'{col}_mean_prev'] = df[f'{col}_mean_prev'].fillna(0)
        df[f'{col}_std_prev'] = df[f'{col}_std_prev'].fillna(0)
        df[f'{col}_trend'] = df[f'{col}_trend'].fillna(0)
    
    return df


def prepare_final_features(df: pd.DataFrame,
                          patient_col: str = 'Patient',
                          exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Prepare final feature list for model input.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    patient_col : str
        Column name for patient ID
    exclude_cols : list, optional
        Columns to exclude from features
        
    Returns:
    --------
    list
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = [
            patient_col, 'target', 'time_to_event', 'event_occurred',
            'measurement_time', 'event_time', 'detection_time', 'is_abn',
            'admission_time', 'adjusted_time'
        ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove columns with all NaN
    feature_cols = [col for col in feature_cols if df[col].notna().sum() > 0]
    
    # Remove datetime and object columns (keep only numeric)
    feature_cols = [col for col in feature_cols 
                   if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    return feature_cols
