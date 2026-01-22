"""
Survival SVM Models using PySurvival
"""

import numpy as np
import pandas as pd
try:
    from pysurvival.models.survival_svm import LinearSurvivalSVM, KernelSurvivalSVM
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False
    LinearSurvivalSVM = None
    KernelSurvivalSVM = None
import joblib
from pathlib import Path


def train_survival_svm(X_train, T_train, E_train, X_val, T_val, E_val,
                      dataset_name, models_to_train=None,
                      C=1.0, kernel='rbf', gamma='scale',
                      save_dir=None):
    """
    Train Survival SVM models (Linear and Kernel)
    
    Note: Survival SVM may not be available in all PySurvival versions
    
    Args:
        X_train: Training features (n_samples, n_features)
        T_train: Training time-to-event (n_samples,)
        E_train: Training event indicator (n_samples,)
        X_val: Validation features
        T_val: Validation time-to-event
        E_val: Validation event indicator
        dataset_name: Name of dataset ('3year' or '10year')
        models_to_train: List of model names ('Linear', 'Kernel', or both)
        C: Regularization parameter
        kernel: Kernel type for Kernel SVM ('rbf', 'polynomial', 'linear')
        gamma: Kernel coefficient for RBF ('scale', 'auto', or float)
        save_dir: Directory to save models
        
    Returns:
        models_dict: Dictionary of trained models
    """
    if not SVM_AVAILABLE:
        print(f"\nWarning: Survival SVM models are not available in this PySurvival version.")
        print(f"Skipping Survival SVM training.")
        return {}
    
    print(f"\n{'='*60}")
    print(f"Training Survival SVM Models for {dataset_name} dataset")
    print(f"{'='*60}")
    
    if models_to_train is None:
        models_to_train = ['Linear', 'Kernel']
    
    trained_models = {}
    
    # Train Linear Survival SVM
    if 'Linear' in models_to_train:
        print(f"\n--- Training Linear Survival SVM ---")
        print(f"  C={C}")
        
        try:
            model = LinearSurvivalSVM(C=C)
            model.fit(X_train, T_train, E_train)
            
            print("Linear Survival SVM training completed!")
            trained_models['LinearSurvivalSVM'] = model
            
            # Save model
            if save_dir is not None:
                save_path = Path(save_dir) / f"linear_survival_svm_{dataset_name}.pkl"
                joblib.dump(model, save_path)
                print(f"Model saved to: {save_path}")
        
        except Exception as e:
            print(f"Error training Linear Survival SVM: {e}")
            import traceback
            traceback.print_exc()
    
    # Train Kernel Survival SVM
    if 'Kernel' in models_to_train:
        print(f"\n--- Training Kernel Survival SVM ({kernel}) ---")
        print(f"  C={C}, kernel={kernel}, gamma={gamma}")
        
        try:
            model = KernelSurvivalSVM(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, T_train, E_train)
            
            print(f"Kernel Survival SVM ({kernel}) training completed!")
            trained_models[f'KernelSurvivalSVM_{kernel}'] = model
            
            # Save model
            if save_dir is not None:
                save_path = Path(save_dir) / f"kernel_survival_svm_{kernel}_{dataset_name}.pkl"
                joblib.dump(model, save_path)
                print(f"Model saved to: {save_path}")
        
        except Exception as e:
            print(f"Error training Kernel Survival SVM: {e}")
            import traceback
            traceback.print_exc()
    
    return trained_models
