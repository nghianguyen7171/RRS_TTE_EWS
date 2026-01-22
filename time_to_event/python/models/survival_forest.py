"""
Survival Forest Models using PySurvival
"""

import numpy as np
import pandas as pd
from pysurvival.models.survival_forest import (
    RandomSurvivalForestModel,
    ExtraSurvivalTreesModel,
    ConditionalSurvivalForestModel
)
import joblib
from pathlib import Path


def train_survival_forests(X_train, T_train, E_train, X_val, T_val, E_val,
                          dataset_name, models_to_train=None,
                          n_estimators=100, max_depth=None, min_samples_split=2,
                          min_samples_leaf=1, save_dir=None):
    """
    Train multiple survival forest models
    
    Args:
        X_train: Training features (n_samples, n_features)
        T_train: Training time-to-event (n_samples,)
        E_train: Training event indicator (n_samples,)
        X_val: Validation features
        T_val: Validation time-to-event
        E_val: Validation event indicator
        dataset_name: Name of dataset ('3year' or '10year')
        models_to_train: List of model names to train (if None, trains all)
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        save_dir: Directory to save models
        
    Returns:
        models_dict: Dictionary of trained models
    """
    print(f"\n{'='*60}")
    print(f"Training Survival Forest Models for {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Define available models
    available_models = {
        'RandomSurvivalForest': RandomSurvivalForestModel,
        'ExtraSurvivalTrees': ExtraSurvivalTreesModel,
        'ConditionalSurvivalForest': ConditionalSurvivalForestModel,
    }
    
    if models_to_train is None:
        models_to_train = list(available_models.keys())
    
    trained_models = {}
    
    for model_name in models_to_train:
        if model_name not in available_models:
            print(f"Warning: {model_name} not available. Skipping...")
            continue
        
        print(f"\n--- Training {model_name} ---")
        print(f"  n_estimators={n_estimators}, max_depth={max_depth}")
        ModelClass = available_models[model_name]
        
        try:
            # Initialize model - check available parameters
            try:
                model = ModelClass(num_trees=n_estimators)
            except:
                # Try alternative parameter names
                try:
                    model = ModelClass(n_estimators=n_estimators)
                except:
                    model = ModelClass()
            
            # Train model
            model.fit(X_train, T_train, E_train)
            
            print(f"{model_name} training completed!")
            
            trained_models[model_name] = model
            
            # Get feature importance if available
            try:
                importances = model.variable_importance
                if importances is not None:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\nTop 10 features by importance:")
                    print(importance_df.head(10).to_string(index=False))
                    
                    # Save importance
                    if save_dir is not None:
                        imp_path = Path(save_dir).parent / "metrics" / f"{model_name.lower()}_importance_{dataset_name}.csv"
                        imp_path.parent.mkdir(parents=True, exist_ok=True)
                        importance_df.to_csv(imp_path, index=False)
            except:
                pass
            
            # Save model
            if save_dir is not None:
                save_path = Path(save_dir) / f"{model_name.lower()}_{dataset_name}.pkl"
                joblib.dump(model, save_path)
                print(f"Model saved to: {save_path}")
        
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return trained_models
