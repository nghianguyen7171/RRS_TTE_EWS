"""
Data validation utilities for time-to-event models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def validate_processed_data(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate processed survival analysis data.
    
    Parameters:
    -----------
    processed_data : dict
        Dictionary from prepare_survival_data
        
    Returns:
    --------
    dict
        Validation report
    """
    report = {
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    X = processed_data['X']
    y_time = processed_data['y_time']
    y_event = processed_data['y_event']
    metadata = processed_data['metadata']
    
    # Check data shapes
    n_samples = len(y_time)
    report['statistics']['n_samples'] = n_samples
    report['statistics']['n_features'] = X.shape[1] if hasattr(X, 'shape') else len(X.columns)
    
    # Check for missing values
    if hasattr(X, 'isnull'):
        missing = X.isnull().sum().sum()
        if missing > 0:
            report['valid'] = False
            report['issues'].append(f"Found {missing} missing values in features")
    
    # Check time-to-event
    if np.any(y_time < 0):
        report['valid'] = False
        report['issues'].append("Found negative time_to_event values")
    
    if np.any(np.isnan(y_time)):
        report['valid'] = False
        report['issues'].append("Found NaN values in time_to_event")
    
    report['statistics']['time_to_event_mean'] = float(np.mean(y_time))
    report['statistics']['time_to_event_median'] = float(np.median(y_time))
    report['statistics']['time_to_event_min'] = float(np.min(y_time))
    report['statistics']['time_to_event_max'] = float(np.max(y_time))
    
    # Check event indicator
    if not np.all(np.isin(y_event, [0, 1])):
        report['valid'] = False
        report['issues'].append("Event indicator contains values other than 0 and 1")
    
    event_rate = np.mean(y_event)
    report['statistics']['event_rate'] = float(event_rate)
    report['statistics']['n_events'] = int(np.sum(y_event))
    report['statistics']['n_censored'] = int(np.sum(1 - y_event))
    
    # Check consistency
    if len(y_time) != len(y_event):
        report['valid'] = False
        report['issues'].append("Mismatch between y_time and y_event lengths")
    
    if hasattr(X, 'shape') and X.shape[0] != len(y_time):
        report['valid'] = False
        report['issues'].append("Mismatch between X and y_time lengths")
    
    return report
