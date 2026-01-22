"""
Data loading utilities for survival analysis
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__).resolve()
    # Go up from python/data_loader.py to time_to_event/
    return current_file.parent.parent


def load_survival_data(dataset_name):
    """
    Load processed survival data for PySurvival
    
    Args:
        dataset_name: '3year' or '10year'
        
    Returns:
        X: DataFrame with features (n_samples, n_features)
        T: numpy array with time-to-event (n_samples,)
        E: numpy array with event indicator (n_samples,)
        metadata: DataFrame with patient IDs and metadata
    """
    project_root = get_project_root()
    base_path = project_root / "data" / f"processed_{dataset_name}"
    
    # Load features
    X_path = base_path / "X_features.csv"
    if not X_path.exists():
        raise FileNotFoundError(f"Features file not found: {X_path}")
    X = pd.read_csv(X_path)
    
    # Load time-to-event
    T_path = base_path / "y_time.npy"
    if not T_path.exists():
        raise FileNotFoundError(f"Time file not found: {T_path}")
    T = np.load(T_path)
    
    # Load event indicator
    E_path = base_path / "y_event.npy"
    if not E_path.exists():
        raise FileNotFoundError(f"Event file not found: {E_path}")
    E = np.load(E_path)
    
    # Load metadata
    metadata_path = base_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    
    # Validate shapes
    n_samples = len(X)
    if len(T) != n_samples or len(E) != n_samples:
        raise ValueError(f"Shape mismatch: X={n_samples}, T={len(T)}, E={len(E)}")
    
    if len(metadata) != n_samples:
        raise ValueError(f"Metadata length {len(metadata)} != {n_samples}")
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Events: {E.sum():,} ({100*E.sum()/n_samples:.2f}%)")
    print(f"  Censored: {(E==0).sum():,} ({100*(E==0).sum()/n_samples:.2f}%)")
    print(f"  Time range: {T.min():.2f} - {T.max():.2f} hours")
    
    return X, T, E, metadata


def prepare_pysurvival_format(X, T, E):
    """
    Convert data to PySurvival format (numpy arrays)
    
    Args:
        X: DataFrame or numpy array with features
        T: numpy array with time-to-event
        E: numpy array with event indicator
        
    Returns:
        X_array: numpy array (n_samples, n_features)
        T_array: numpy array (n_samples,)
        E_array: numpy array (n_samples,)
    """
    # Convert X to numpy if DataFrame
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.array(X)
    
    # Ensure T and E are numpy arrays
    T_array = np.array(T, dtype=np.float64)
    E_array = np.array(E, dtype=np.int32)
    
    # Validate
    if len(T_array) != len(X_array) or len(E_array) != len(X_array):
        raise ValueError("X, T, E must have same length")
    
    # Check for invalid values
    if np.any(T_array < 0):
        raise ValueError("Time values must be non-negative")
    if np.any((E_array != 0) & (E_array != 1)):
        raise ValueError("Event indicator must be 0 or 1")
    
    return X_array, T_array, E_array


def get_patient_ids(metadata):
    """
    Extract unique patient IDs from metadata
    
    Args:
        metadata: DataFrame with 'patient_id' column
        
    Returns:
        patient_ids: numpy array of unique patient IDs
    """
    if 'patient_id' not in metadata.columns:
        raise ValueError("metadata must contain 'patient_id' column")
    
    patient_ids = metadata['patient_id'].unique()
    return patient_ids


def get_feature_names(X):
    """
    Get feature names from X
    
    Args:
        X: DataFrame or array
        
    Returns:
        feature_names: list of feature names
    """
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    else:
        return [f"feature_{i}" for i in range(X.shape[1])]
