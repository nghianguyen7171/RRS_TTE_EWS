"""
Evaluation metrics for survival models
"""

import numpy as np
import pandas as pd
from pysurvival.utils.metrics import concordance_index
from scipy import stats


def calculate_c_index(model, X, T, E):
    """
    Calculate concordance index (C-index) for survival model
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix (n_samples, n_features)
        T: Time-to-event array (n_samples,)
        E: Event indicator array (n_samples,)
        
    Returns:
        c_index: Concordance index (0.5 = random, 1.0 = perfect)
    """
    try:
        # PySurvival's concordance_index function
        c_index = concordance_index(model, X, T, E)
        return c_index
    except Exception as e:
        print(f"Error calculating C-index: {e}")
        return None


def calculate_brier_score(model, X, T, E, time_points=None):
    """
    Calculate Integrated Brier Score (IBS) for survival model
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix (n_samples, n_features)
        T: Time-to-event array (n_samples,)
        E: Event indicator array (n_samples,)
        time_points: Array of time points for evaluation (if None, uses observed times)
        
    Returns:
        ibs: Integrated Brier Score (lower is better, 0 = perfect)
        brier_scores: Brier scores at each time point
    """
    try:
        from pysurvival.utils.metrics import brier_score
        
        if time_points is None:
            # Use observed event times
            time_points = np.unique(T[E == 1])
            if len(time_points) == 0:
                time_points = np.linspace(T.min(), T.max(), 10)
        
        # Calculate Brier score at each time point
        brier_scores = []
        for t in time_points:
            try:
                bs = brier_score(model, X, T, E, t=t)
                brier_scores.append(bs)
            except:
                continue
        
        if len(brier_scores) == 0:
            return None, None
        
        # Integrated Brier Score (average)
        ibs = np.mean(brier_scores)
        
        return ibs, np.array(brier_scores)
    except Exception as e:
        print(f"Error calculating Brier Score: {e}")
        return None, None


def calculate_calibration(model, X, T, E, n_bins=10, time_point=None):
    """
    Calculate calibration metrics (observed vs predicted survival)
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix (n_samples, n_features)
        T: Time-to-event array (n_samples,)
        E: Event indicator array (n_samples,)
        n_bins: Number of bins for calibration
        time_point: Time point for calibration (if None, uses median time)
        
    Returns:
        calibration_df: DataFrame with calibration results
    """
    try:
        if time_point is None:
            time_point = np.median(T[E == 1]) if (E == 1).sum() > 0 else np.median(T)
        
        # Predict survival probabilities
        predictions = model.predict_survival(X, t=time_point)
        
        # Create bins based on predicted probabilities
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate observed survival in each bin
        calibration_results = []
        for i in range(n_bins):
            mask = (bin_indices == i)
            if mask.sum() == 0:
                continue
            
            bin_predictions = predictions[mask]
            bin_times = T[mask]
            bin_events = E[mask]
            
            # Observed survival at time_point
            observed_survival = ((bin_times > time_point) | 
                                ((bin_times <= time_point) & (bin_events == 0))).mean()
            
            # Predicted survival (mean in bin)
            predicted_survival = bin_predictions.mean()
            
            calibration_results.append({
                'bin': i,
                'n_samples': mask.sum(),
                'predicted_survival': predicted_survival,
                'observed_survival': observed_survival,
                'calibration_error': abs(predicted_survival - observed_survival)
            })
        
        calibration_df = pd.DataFrame(calibration_results)
        
        # Calculate ECE (Expected Calibration Error)
        if len(calibration_df) > 0:
            ece = (calibration_df['n_samples'] * calibration_df['calibration_error']).sum() / len(X)
        else:
            ece = None
        
        return calibration_df, ece
    except Exception as e:
        print(f"Error calculating calibration: {e}")
        return None, None


def risk_stratification(model, X, T, E, n_groups=4):
    """
    Perform risk stratification and calculate event rates by risk group
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix (n_samples, n_features)
        T: Time-to-event array (n_samples,)
        E: Event indicator array (n_samples,)
        n_groups: Number of risk groups (default: 4 quartiles)
        
    Returns:
        stratification_df: DataFrame with risk group statistics
    """
    try:
        # Get risk scores (higher = higher risk)
        # For Cox models, use negative log survival probability
        # For other models, use predicted risk
        try:
            risk_scores = model.predict_risk(X)
        except:
            # Fallback: use negative log survival at median time
            median_time = np.median(T[E == 1]) if (E == 1).sum() > 0 else np.median(T)
            survival_probs = model.predict_survival(X, t=median_time)
            risk_scores = -np.log(survival_probs + 1e-10)
        
        # Create risk groups (quartiles)
        percentiles = np.linspace(0, 100, n_groups + 1)
        risk_thresholds = np.percentile(risk_scores, percentiles)
        
        # Assign groups
        groups = np.digitize(risk_scores, risk_thresholds[1:])
        groups = np.clip(groups, 0, n_groups - 1)
        
        # Calculate statistics per group
        stratification_results = []
        for i in range(n_groups):
            mask = (groups == i)
            if mask.sum() == 0:
                continue
            
            group_times = T[mask]
            group_events = E[mask]
            group_risk_scores = risk_scores[mask]
            
            stratification_results.append({
                'risk_group': i,
                'group_label': f'Q{i+1}',
                'n_samples': mask.sum(),
                'mean_risk_score': group_risk_scores.mean(),
                'median_risk_score': np.median(group_risk_scores),
                'n_events': group_events.sum(),
                'event_rate': group_events.sum() / mask.sum(),
                'median_time': np.median(group_times),
            })
        
        stratification_df = pd.DataFrame(stratification_results)
        
        return stratification_df, risk_scores, groups
    except Exception as e:
        print(f"Error in risk stratification: {e}")
        return None, None, None


def calculate_all_metrics(model, X_train, T_train, E_train, 
                         X_val, T_val, E_val, 
                         X_test, T_test, E_test,
                         dataset_name, model_name):
    """
    Calculate all evaluation metrics for a model
    
    Returns:
        metrics_dict: Dictionary with all metrics
    """
    metrics = {
        'dataset': dataset_name,
        'model': model_name,
    }
    
    # C-index on all sets
    print(f"  Calculating C-index...")
    c_train = calculate_c_index(model, X_train, T_train, E_train)
    c_val = calculate_c_index(model, X_val, T_val, E_val)
    c_test = calculate_c_index(model, X_test, T_test, E_test)
    
    metrics['c_index_train'] = c_train
    metrics['c_index_val'] = c_val
    metrics['c_index_test'] = c_test
    
    # Brier Score
    print(f"  Calculating Brier Score...")
    ibs_val, _ = calculate_brier_score(model, X_val, T_val, E_val)
    ibs_test, _ = calculate_brier_score(model, X_test, T_test, E_test)
    
    metrics['ibs_val'] = ibs_val
    metrics['ibs_test'] = ibs_test
    
    # Calibration
    print(f"  Calculating calibration...")
    cal_val, ece_val = calculate_calibration(model, X_val, T_val, E_val)
    cal_test, ece_test = calculate_calibration(model, X_test, T_test, E_test)
    
    metrics['ece_val'] = ece_val
    metrics['ece_test'] = ece_test
    
    # Risk stratification
    print(f"  Performing risk stratification...")
    strat_val, risk_scores_val, groups_val = risk_stratification(model, X_val, T_val, E_val)
    strat_test, risk_scores_test, groups_test = risk_stratification(model, X_test, T_test, E_test)
    
    metrics['stratification_val'] = strat_val
    metrics['stratification_test'] = strat_test
    metrics['risk_scores_val'] = risk_scores_val
    metrics['risk_scores_test'] = risk_scores_test
    metrics['groups_val'] = groups_val
    metrics['groups_test'] = groups_test
    
    return metrics
