"""
Patient-level data splitting to avoid data leakage
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratify_by_event(patient_ids, metadata):
    """
    Get event status for each patient (whether patient ever had an event)
    
    Args:
        patient_ids: array of patient IDs
        metadata: DataFrame with 'patient_id' and 'event_occurred' columns
        
    Returns:
        patient_events: array of event status (1 if patient had event, 0 otherwise)
    """
    # Get event status per patient (max event_occurred for each patient)
    patient_event_status = metadata.groupby('patient_id')['event_occurred'].max().reset_index()
    patient_event_dict = dict(zip(patient_event_status['patient_id'], 
                                   patient_event_status['event_occurred']))
    
    # Map patient IDs to event status
    patient_events = np.array([patient_event_dict.get(pid, 0) for pid in patient_ids])
    
    return patient_events


def patient_level_split(X, T, E, metadata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data at patient level to avoid data leakage
    
    Args:
        X: DataFrame or array with features (n_samples, n_features)
        T: array with time-to-event (n_samples,)
        E: array with event indicator (n_samples,)
        metadata: DataFrame with 'patient_id' column
        train_ratio: proportion for training set
        val_ratio: proportion for validation set
        test_ratio: proportion for test set
        random_state: random seed
        
    Returns:
        Dictionary with train/val/test splits for X, T, E, and metadata
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # Get unique patient IDs
    patient_ids = metadata['patient_id'].unique()
    n_patients = len(patient_ids)
    
    print(f"Splitting {n_patients:,} patients into train/val/test sets...")
    
    # Get event status for stratification
    patient_events = stratify_by_event(patient_ids, metadata)
    
    # First split: train vs (val + test)
    train_patients, temp_patients, train_events, temp_events = train_test_split(
        patient_ids, patient_events,
        test_size=(val_ratio + test_ratio),
        stratify=patient_events,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients, val_events, test_events = train_test_split(
        temp_patients, temp_events,
        test_size=(1 - val_size),
        stratify=temp_events,
        random_state=random_state
    )
    
    # Create masks for each split
    train_mask = metadata['patient_id'].isin(train_patients)
    val_mask = metadata['patient_id'].isin(val_patients)
    test_mask = metadata['patient_id'].isin(test_patients)
    
    # Split data
    if isinstance(X, pd.DataFrame):
        X_train = X[train_mask].reset_index(drop=True)
        X_val = X[val_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
    else:
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
    
    T_train = T[train_mask]
    T_val = T[val_mask]
    T_test = T[test_mask]
    
    E_train = E[train_mask]
    E_val = E[val_mask]
    E_test = E[test_mask]
    
    metadata_train = metadata[train_mask].reset_index(drop=True)
    metadata_val = metadata[val_mask].reset_index(drop=True)
    metadata_test = metadata[test_mask].reset_index(drop=True)
    
    # Print statistics
    print(f"\nTrain set:")
    print(f"  Patients: {len(train_patients):,}")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Events: {E_train.sum():,} ({100*E_train.sum()/len(E_train):.2f}%)")
    
    print(f"\nValidation set:")
    print(f"  Patients: {len(val_patients):,}")
    print(f"  Samples: {len(X_val):,}")
    print(f"  Events: {E_val.sum():,} ({100*E_val.sum()/len(E_val):.2f}%)")
    
    print(f"\nTest set:")
    print(f"  Patients: {len(test_patients):,}")
    print(f"  Samples: {len(X_test):,}")
    print(f"  Events: {E_test.sum():,} ({100*E_test.sum()/len(E_test):.2f}%)")
    
    return {
        'X_train': X_train, 'T_train': T_train, 'E_train': E_train, 'metadata_train': metadata_train,
        'X_val': X_val, 'T_val': T_val, 'E_val': E_val, 'metadata_val': metadata_val,
        'X_test': X_test, 'T_test': T_test, 'E_test': E_test, 'metadata_test': metadata_test,
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients,
    }
