"""
Parametric Survival Models using PySurvival
"""

import numpy as np
import pandas as pd
from pysurvival.models.parametric import (
    ExponentialModel,
    WeibullModel,
    GompertzModel,
    LogLogisticModel,
    LogNormalModel
)
import joblib
from pathlib import Path


def train_parametric_models(X_train, T_train, E_train, X_val, T_val, E_val,
                            dataset_name, models_to_train=None,
                            save_dir=None):
    """
    Train multiple parametric survival models and compare them
    
    Args:
        X_train: Training features (n_samples, n_features)
        T_train: Training time-to-event (n_samples,)
        E_train: Training event indicator (n_samples,)
        X_val: Validation features
        T_val: Validation time-to-event
        E_val: Validation event indicator
        dataset_name: Name of dataset ('3year' or '10year')
        models_to_train: List of model names to train (if None, trains all)
        save_dir: Directory to save models
        
    Returns:
        models_dict: Dictionary of trained models
        comparison_df: DataFrame comparing models (AIC, BIC)
    """
    print(f"\n{'='*60}")
    print(f"Training Parametric Models for {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Define available models
    available_models = {
        'Exponential': ExponentialModel,
        'Weibull': WeibullModel,
        'Gompertz': GompertzModel,
        'LogLogistic': LogLogisticModel,
        'LogNormal': LogNormalModel,
    }
    
    if models_to_train is None:
        models_to_train = list(available_models.keys())
    
    trained_models = {}
    model_comparison = []
    
    for model_name in models_to_train:
        if model_name not in available_models:
            print(f"Warning: {model_name} not available. Skipping...")
            continue
        
        print(f"\n--- Training {model_name} ---")
        ModelClass = available_models[model_name]
        
        try:
            # Initialize and train model
            model = ModelClass()
            model.fit(X_train, T_train, E_train)
            
            print(f"{model_name} training completed!")
            
            # Get AIC and BIC if available
            try:
                aic = model.AIC
                bic = model.BIC
            except:
                aic = None
                bic = None
            
            trained_models[model_name] = model
            
            model_comparison.append({
                'model': model_name,
                'AIC': aic,
                'BIC': bic,
                'status': 'success'
            })
            
            # Save model
            if save_dir is not None:
                save_path = Path(save_dir) / f"parametric_{model_name.lower()}_{dataset_name}.pkl"
                joblib.dump(model, save_path)
                print(f"Model saved to: {save_path}")
        
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            model_comparison.append({
                'model': model_name,
                'AIC': None,
                'BIC': None,
                'status': f'error: {str(e)}'
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(model_comparison)
    
    if len(comparison_df) > 0:
        print(f"\n{'='*60}")
        print("Model Comparison (AIC/BIC - lower is better):")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        if save_dir is not None:
            comp_path = Path(save_dir).parent / "metrics" / f"parametric_comparison_{dataset_name}.csv"
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(comp_path, index=False)
            print(f"\nComparison saved to: {comp_path}")
    
    return trained_models, comparison_df
