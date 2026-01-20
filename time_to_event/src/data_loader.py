"""
Data loading utilities for time-to-event (survival analysis) models.
"""

import sys
import os
# Add parent project src to path
parent_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Import from parent project
import importlib.util
spec = importlib.util.spec_from_file_location("parent_data_loader", 
    os.path.join(parent_src, "data_loader.py"))
parent_data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_data_loader)
load_3year_base = parent_data_loader.load_3year_dataset
load_10year_base = parent_data_loader.load_10year_dataset


def impute_event_time_from_target(
    df: pd.DataFrame,
    patient_col: str = "Patient",
    target_col: str = "target",
    measurement_time_col: str = "measurement_time",
    event_time_col: str = "event_time",
) -> pd.DataFrame:
    """
    Impute event_time from target column and measurement_time.
    
    For patients with target=1, event_time is set to the last measurement_time
    where target=1 for that patient.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with Patient, target, measurement_time, and event_time columns
    patient_col : str
        Column name for patient ID
    target_col : str
        Column name for target (1 = deterioration)
    measurement_time_col : str
        Column name for measurement time
    event_time_col : str
        Column name for event time (to be imputed)
        
    Returns:
    --------
    pd.DataFrame
        Dataset with imputed event_time
    """
    df = df.copy()
    
    # Ensure datetime
    df[measurement_time_col] = pd.to_datetime(df[measurement_time_col], errors="coerce")
    if event_time_col in df.columns:
        df[event_time_col] = pd.to_datetime(df[event_time_col], errors="coerce")
    else:
        df[event_time_col] = pd.NaT
    
    # Find patients with target=1
    positive_mask = df[target_col] == 1
    positive_patients = df.loc[positive_mask, patient_col].dropna().unique()
    
    imputed_rows = 0
    # Precompute for speed: group by patient
    grouped = df.groupby(patient_col)
    
    for patient_id in positive_patients:
        try:
            patient_data = grouped.get_group(patient_id)
        except KeyError:
            continue
        
        # Rows where target == 1 for this patient
        positive_rows = patient_data[patient_data[target_col] == 1]
        if positive_rows.empty:
            continue
        
        # Last measurement_time where target == 1
        last_event_time = positive_rows[measurement_time_col].max()
        if pd.isna(last_event_time):
            continue
        
        # Impute event_time for this patient where event_time is missing
        mask = (df[patient_col] == patient_id) & (df[event_time_col].isna())
        df.loc[mask, event_time_col] = last_event_time
        imputed_rows += mask.sum()
    
    return df


def load_3year_dataset(data_path: Optional[str] = None, impute_event_time: bool = True) -> pd.DataFrame:
    """
    Load the 3-year dataset for time-to-event analysis.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the 3-year dataset CSV file. If None, uses default path.
    impute_event_time : bool
        Whether to impute missing event times from target column
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset with proper data types and imputed event times
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'CNUH_3Y.csv')
        data_path = os.path.abspath(data_path)
    else:
        # Path provided - use as-is if absolute, otherwise make absolute
        original_path = data_path
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)
        # Verify the path is correct
        if not os.path.exists(data_path):
            # Try the original path as-is
            if os.path.exists(original_path):
                data_path = original_path
            else:
                raise FileNotFoundError(
                    f"Data file not found: {data_path}\n"
                    f"Original path: {original_path}\n"
                    f"Current working directory: {os.getcwd()}"
                )
    
    # Final verification
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}\nCurrent working directory: {os.getcwd()}")
    
    df = load_3year_base(data_path)
    
    if impute_event_time:
        df = impute_event_time_from_target(df)
    
    return df


def load_10year_dataset(data_path: Optional[str] = None, impute_event_time: bool = True) -> pd.DataFrame:
    """
    Load the 10-year dataset for time-to-event analysis.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the 10-year dataset CSV file. If None, uses default path.
    impute_event_time : bool
        Whether to impute missing event times from target column
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset with proper data types and imputed event times
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '10yrs_proc.csv')
    
    # Convert to absolute path if not already absolute
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(data_path)
    
    df = load_10year_base(data_path)
    
    if impute_event_time:
        df = impute_event_time_from_target(df)
    
    return df


def validate_temporal_data(df: pd.DataFrame,
                          patient_col: str = 'Patient',
                          measurement_time_col: str = 'measurement_time',
                          event_time_col: str = 'event_time',
                          target_col: str = 'target') -> Dict[str, Any]:
    """
    Validate temporal data structure for time-to-event analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    patient_col : str
        Column name for patient ID
    measurement_time_col : str
        Column name for measurement time
    event_time_col : str
        Column name for event time
    target_col : str
        Column name for target
        
    Returns:
    --------
    dict
        Validation summary
    """
    validation = {
        'total_rows': len(df),
        'unique_patients': df[patient_col].nunique(),
        'measurement_time_complete': df[measurement_time_col].notna().sum(),
        'event_time_complete': df[event_time_col].notna().sum(),
        'positive_cases': (df[target_col] == 1).sum(),
        'positive_cases_with_event_time': 0,
        'patients_with_events': 0,
        'patients_without_events': 0,
        'temporal_consistency': True,
        'issues': []
    }
    
    # Check positive cases with event time
    positive_cases = df[df[target_col] == 1]
    validation['positive_cases_with_event_time'] = positive_cases[event_time_col].notna().sum()
    
    # Check patients with/without events
    patients_with_events = df[df[target_col] == 1][patient_col].unique()
    validation['patients_with_events'] = len(patients_with_events)
    validation['patients_without_events'] = df[patient_col].nunique() - len(patients_with_events)
    
    # Check temporal consistency (measurement_time should be <= event_time for positive cases)
    if len(positive_cases) > 0:
        positive_with_event = positive_cases[
            positive_cases[event_time_col].notna()
        ]
        if len(positive_with_event) > 0:
            invalid = positive_with_event[
                positive_with_event[measurement_time_col] > positive_with_event[event_time_col]
            ]
            if len(invalid) > 0:
                validation['temporal_consistency'] = False
                validation['issues'].append(
                    f"{len(invalid)} measurements occur after event time"
                )
    
    return validation
