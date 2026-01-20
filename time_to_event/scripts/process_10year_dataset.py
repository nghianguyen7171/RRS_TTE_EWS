#!/usr/bin/env python3
"""
Process 10-year dataset for time-to-event (survival analysis) models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.data_loader import load_10year_dataset, validate_temporal_data
from src.preprocessing import (
    handle_missing_values, detect_outliers_iqr, handle_outliers,
    normalize_features, create_temporal_features
)
from src.survival_labels import prepare_survival_data
from src.feature_engineering import (
    get_feature_groups, create_interaction_features,
    create_aggregated_features, prepare_final_features
)
from src.validation import validate_processed_data

def main():
    print("=" * 80)
    print("PROCESSING 10-YEAR DATASET FOR TIME-TO-EVENT MODELS")
    print("=" * 80)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = Path(script_dir).parent
    data_dir = base_dir / "data" / "processed_10year"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = base_dir / "results" / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get absolute path to parent project data directory
    project_root = base_dir.parent.resolve()
    raw_data_path = str(project_root / "data" / "10yrs_proc.csv")
    
    # Verify path before proceeding
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Data file not found at: {raw_data_path}")
    
    # Step 1: Load data
    print("\n" + "-" * 80)
    print("Step 1: Loading data")
    print("-" * 80)
    df = load_10year_dataset(raw_data_path, impute_event_time=True)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Validate temporal data
    temporal_validation = validate_temporal_data(df)
    print(f"\nTemporal validation:")
    print(f"  Unique patients: {temporal_validation['unique_patients']:,}")
    print(f"  Positive cases: {temporal_validation['positive_cases']:,}")
    print(f"  Positive cases with event_time: {temporal_validation['positive_cases_with_event_time']:,}")
    print(f"  Temporal consistency: {temporal_validation['temporal_consistency']}")
    
    # Step 2: Preprocessing
    print("\n" + "-" * 80)
    print("Step 2: Preprocessing")
    print("-" * 80)
    
    # Get feature groups
    feature_groups = get_feature_groups(df)
    all_features = (feature_groups['vital_signs'] + feature_groups['lab_values'] + 
                   feature_groups['demographics'])
    
    # Handle missing values
    print("Handling missing values...")
    df = handle_missing_values(df, all_features, strategy='median')
    
    # Detect outliers
    print("Detecting outliers...")
    outlier_info = detect_outliers_iqr(df, all_features)
    high_outlier_features = [k for k, v in outlier_info.items() if v['percentage'] > 10]
    if high_outlier_features:
        print(f"  Features with >10% outliers: {len(high_outlier_features)}")
    
    # Handle outliers (clip)
    print("Handling outliers (clipping)...")
    df = handle_outliers(df, all_features, method='clip', outlier_info=outlier_info)
    
    # Create temporal features
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    # Step 3: Feature engineering
    print("\n" + "-" * 80)
    print("Step 3: Feature engineering")
    print("-" * 80)
    
    # Create interaction features
    print("Creating interaction features...")
    df = create_interaction_features(df, feature_groups)
    
    # Create aggregated features
    print("Creating aggregated features...")
    df = create_aggregated_features(df, feature_cols=feature_groups['vital_signs'][:3])
    
    # Step 4: Prepare survival data
    print("\n" + "-" * 80)
    print("Step 4: Preparing survival labels")
    print("-" * 80)
    
    # Get final feature list
    final_features = prepare_final_features(df)
    print(f"Final feature count: {len(final_features)}")
    
    # Prepare survival data
    processed_data = prepare_survival_data(df, feature_cols=final_features)
    
    # Step 5: Normalize features
    print("\n" + "-" * 80)
    print("Step 5: Normalizing features")
    print("-" * 80)
    
    X_normalized, scaler_info = normalize_features(
        processed_data['X'], final_features, method='standardize'
    )
    processed_data['X'] = X_normalized
    
    # Step 6: Validation
    print("\n" + "-" * 80)
    print("Step 6: Validating processed data")
    print("-" * 80)
    
    validation_report = validate_processed_data(processed_data)
    
    if validation_report['valid']:
        print("✓ Data validation passed")
    else:
        print("✗ Data validation failed:")
        for issue in validation_report['issues']:
            print(f"  - {issue}")
    
    print(f"\nStatistics:")
    for key, value in validation_report['statistics'].items():
        print(f"  {key}: {value}")
    
    # Step 7: Save processed data
    print("\n" + "-" * 80)
    print("Step 7: Saving processed data")
    print("-" * 80)
    
    # Save feature matrix
    processed_data['X'].to_csv(data_dir / "X_features.csv", index=False)
    print(f"Saved features to: {data_dir / 'X_features.csv'}")
    
    # Save labels
    np.save(data_dir / "y_time.npy", processed_data['y_time'])
    np.save(data_dir / "y_event.npy", processed_data['y_event'])
    print(f"Saved labels to: {data_dir / 'y_time.npy'} and {data_dir / 'y_event.npy'}")
    
    # Save metadata
    metadata_df = pd.DataFrame({
        'patient_id': processed_data['metadata']['patient_ids'],
        'measurement_time': processed_data['metadata']['measurement_times'],
        'event_time': processed_data['metadata']['event_times'],
        'time_to_event': processed_data['metadata']['time_to_event'],
        'event_occurred': processed_data['metadata']['event_occurred']
    })
    metadata_df.to_csv(data_dir / "metadata.csv", index=False)
    print(f"Saved metadata to: {data_dir / 'metadata.csv'}")
    
    # Save processing report
    report = {
        'validation': validation_report,
        'feature_names': processed_data['feature_names'],
        'scaler_info': scaler_info,
        'outlier_info': {k: v for k, v in list(outlier_info.items())[:10]}  # Sample
    }
    
    with open(results_dir / "processing_report_10year.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved processing report to: {results_dir / 'processing_report_10year.json'}")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
