"""
Survival analysis label creation for time-to-event models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def calculate_time_to_event(df: pd.DataFrame,
                           patient_col: str = 'Patient',
                           measurement_time_col: str = 'measurement_time',
                           event_time_col: str = 'event_time',
                           target_col: str = 'target') -> pd.DataFrame:
    """
    Calculate time-to-event for each measurement.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
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
    pd.DataFrame
        Dataset with time_to_event column added
    """
    df = df.copy()
    
    # Ensure datetime
    df[measurement_time_col] = pd.to_datetime(df[measurement_time_col], errors='coerce')
    df[event_time_col] = pd.to_datetime(df[event_time_col], errors='coerce')
    
    # Calculate time-to-event for each row
    def calc_ttl(row):
        if pd.isna(row[event_time_col]):
            # No event time - will be handled as censored
            return None
        else:
            # Time until event (in hours)
            time_diff = (row[event_time_col] - row[measurement_time_col]).total_seconds() / 3600
            return max(0, time_diff)  # Can't be negative
    
    df['time_to_event'] = df.apply(calc_ttl, axis=1)
    
    return df


def handle_censoring(df: pd.DataFrame,
                    patient_col: str = 'Patient',
                    measurement_time_col: str = 'measurement_time',
                    event_time_col: str = 'event_time',
                    target_col: str = 'target',
                    time_to_event_col: str = 'time_to_event') -> pd.DataFrame:
    """
    Handle censoring for patients without events.
    
    For censored patients (target=0), set time_to_event to observation time
    (time from first to last measurement, or until discharge).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    patient_col : str
        Column name for patient ID
    measurement_time_col : str
        Column name for measurement time
    event_time_col : str
        Column name for event time
    target_col : str
        Column name for target
    time_to_event_col : str
        Column name for time-to-event
        
    Returns:
    --------
    pd.DataFrame
        Dataset with censoring handled
    """
    df = df.copy()
    
    # Get censored patients (those without events)
    censored_patients = df[df[target_col] == 0][patient_col].unique()
    
    for patient_id in censored_patients:
        patient_data = df[df[patient_col] == patient_id]
        
        if len(patient_data) == 0:
            continue
        
        # Calculate observation time (from first to last measurement)
        max_time = patient_data[measurement_time_col].max()
        min_time = patient_data[measurement_time_col].min()
        
        if pd.isna(max_time) or pd.isna(min_time):
            observation_time = 0
        else:
            observation_time = (max_time - min_time).total_seconds() / 3600
        
        # Set time_to_event for censored patients
        mask = (df[patient_col] == patient_id) & (df[target_col] == 0)
        df.loc[mask, time_to_event_col] = observation_time
    
    return df


def create_event_indicator(df: pd.DataFrame,
                          target_col: str = 'target',
                          event_indicator_col: str = 'event_occurred') -> pd.DataFrame:
    """
    Create event indicator (binary: 0=censored, 1=event occurred).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Column name for target
    event_indicator_col : str
        Column name for event indicator
        
    Returns:
    --------
    pd.DataFrame
        Dataset with event indicator column added
    """
    df = df.copy()
    df[event_indicator_col] = df[target_col].astype(int)
    return df


def filter_invalid_rows(df: pd.DataFrame,
                       measurement_time_col: str = 'measurement_time',
                       event_time_col: str = 'event_time',
                       time_to_event_col: str = 'time_to_event') -> pd.DataFrame:
    """
    Filter out invalid rows (measurements after event time).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    measurement_time_col : str
        Column name for measurement time
    event_time_col : str
        Column name for event time
    time_to_event_col : str
        Column name for time-to-event
        
    Returns:
    --------
    pd.DataFrame
        Dataset with invalid rows removed
    """
    df = df.copy()
    
    # Remove rows where measurement_time > event_time (for positive cases)
    mask = df[event_time_col].notna()
    invalid = df[mask & (df[measurement_time_col] > df[event_time_col])]
    
    if len(invalid) > 0:
        print(f"Removing {len(invalid)} rows where measurement_time > event_time")
        df = df[~(mask & (df[measurement_time_col] > df[event_time_col]))]
    
    # Remove rows with negative time_to_event
    if time_to_event_col in df.columns:
        invalid_ttl = df[df[time_to_event_col] < 0]
        if len(invalid_ttl) > 0:
            print(f"Removing {len(invalid_ttl)} rows with negative time_to_event")
            df = df[df[time_to_event_col] >= 0]
    
    return df


def prepare_survival_data(df: pd.DataFrame,
                         patient_col: str = 'Patient',
                         measurement_time_col: str = 'measurement_time',
                         event_time_col: str = 'event_time',
                         target_col: str = 'target',
                         feature_cols: Optional[list] = None) -> Dict[str, Any]:
    """
    Complete pipeline to prepare survival analysis data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    patient_col : str
        Column name for patient ID
    measurement_time_col : str
        Column name for measurement time
    event_time_col : str
        Column name for event time
    target_col : str
        Column name for target
    feature_cols : list, optional
        List of feature columns. If None, auto-detect.
        
    Returns:
    --------
    dict
        Dictionary with:
        - 'X': Feature matrix
        - 'y_time': Time-to-event values
        - 'y_event': Event indicator
        - 'metadata': Patient IDs, times, etc.
        - 'data': Full processed dataframe
    """
    df = df.copy()
    
    # Step 1: Calculate time-to-event
    df = calculate_time_to_event(df, patient_col, measurement_time_col, 
                                event_time_col, target_col)
    
    # Step 2: Handle censoring
    df = handle_censoring(df, patient_col, measurement_time_col, 
                         event_time_col, target_col)
    
    # Step 3: Create event indicator
    df = create_event_indicator(df, target_col)
    
    # Step 4: Filter invalid rows
    df = filter_invalid_rows(df, measurement_time_col, event_time_col)
    
    # Step 5: Prepare features
    if feature_cols is None:
        # Auto-detect feature columns (exclude metadata columns)
        exclude_cols = [patient_col, measurement_time_col, event_time_col, 
                       target_col, 'time_to_event', 'event_occurred',
                       'admission_time', 'is_abn']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove rows with missing time_to_event
    df = df[df['time_to_event'].notna()]
    
    # Prepare outputs
    X = df[feature_cols].copy()
    y_time = df['time_to_event'].values
    y_event = df['event_occurred'].values
    
    metadata = {
        'patient_ids': df[patient_col].values,
        'measurement_times': df[measurement_time_col].values,
        'event_times': df[event_time_col].values,
        'time_to_event': y_time,
        'event_occurred': y_event
    }
    
    return {
        'X': X,
        'y_time': y_time,
        'y_event': y_event,
        'metadata': metadata,
        'data': df,
        'feature_names': feature_cols
    }
