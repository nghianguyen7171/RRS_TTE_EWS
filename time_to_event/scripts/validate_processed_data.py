#!/usr/bin/env python3
"""
Validate processed time-to-event datasets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pathlib import Path
from src.validation import validate_processed_data

def load_processed_data(data_dir: Path):
    """Load processed data from directory."""
    X = pd.read_csv(data_dir / "X_features.csv")
    y_time = np.load(data_dir / "y_time.npy")
    y_event = np.load(data_dir / "y_event.npy")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    
    processed_data = {
        'X': X,
        'y_time': y_time,
        'y_event': y_event,
        'metadata': {
            'patient_ids': metadata['patient_id'].values,
            'measurement_times': pd.to_datetime(metadata['measurement_time']).values,
            'event_times': pd.to_datetime(metadata['event_time'], errors='coerce').values,
            'time_to_event': metadata['time_to_event'].values,
            'event_occurred': metadata['event_occurred'].values
        },
        'feature_names': list(X.columns)
    }
    
    return processed_data

def main():
    base_dir = Path(__file__).parent.parent
    
    print("=" * 80)
    print("VALIDATING PROCESSED TIME-TO-EVENT DATASETS")
    print("=" * 80)
    
    # Validate 3-year dataset
    print("\n" + "-" * 80)
    print("3-Year Dataset")
    print("-" * 80)
    data_dir_3year = base_dir / "data" / "processed_3year"
    if data_dir_3year.exists():
        processed_data = load_processed_data(data_dir_3year)
        validation = validate_processed_data(processed_data)
        
        if validation['valid']:
            print("✓ Validation passed")
        else:
            print("✗ Validation failed:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        print("\nStatistics:")
        for key, value in validation['statistics'].items():
            print(f"  {key}: {value}")
    else:
        print("Processed data not found. Run process_3year_dataset.py first.")
    
    # Validate 10-year dataset
    print("\n" + "-" * 80)
    print("10-Year Dataset")
    print("-" * 80)
    data_dir_10year = base_dir / "data" / "processed_10year"
    if data_dir_10year.exists():
        processed_data = load_processed_data(data_dir_10year)
        validation = validate_processed_data(processed_data)
        
        if validation['valid']:
            print("✓ Validation passed")
        else:
            print("✗ Validation failed:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        print("\nStatistics:")
        for key, value in validation['statistics'].items():
            print(f"  {key}: {value}")
    else:
        print("Processed data not found. Run process_10year_dataset.py first.")

if __name__ == "__main__":
    main()
