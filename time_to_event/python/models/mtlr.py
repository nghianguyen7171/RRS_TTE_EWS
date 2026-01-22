"""
Multi-Task Logistic Regression (MTLR) Models using PySurvival
"""

import numpy as np
import pandas as pd
from pysurvival.models.multi_task import LinearMultiTaskModel, NeuralMultiTaskModel
import joblib
from pathlib import Path


def train_mtlr(X_train, T_train, E_train, X_val, T_val, E_val,
               dataset_name, model_name="MTLR",
               init_method='glorot_uniform', optimizer='adam', lr=8e-4,
               save_dir=None):
    """
    Train Linear Multi-Task Logistic Regression model
    
    Args:
        X_train: Training features (n_samples, n_features)
        T_train: Training time-to-event (n_samples,)
        E_train: Training event indicator (n_samples,)
        X_val: Validation features
        T_val: Validation time-to-event
        E_val: Validation event indicator
        dataset_name: Name of dataset ('3year' or '10year')
        model_name: Name for the model
        init_method: Weight initialization method
        optimizer: Optimizer ('adam', 'sgd', etc.)
        lr: Learning rate
        save_dir: Directory to save model
        
    Returns:
        model: Trained MTLR model
        results: Dictionary with model information
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Linear) for {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Initialize model
    model = LinearMultiTaskModel()
    
    # Train model
    print(f"Training with optimizer={optimizer}, lr={lr}...")
    model.fit(
        X=X_train,
        T=T_train,
        E=E_train,
        init_method=init_method,
        optimizer=optimizer,
        lr=lr
    )
    
    print("Training completed!")
    
    # Save model
    if save_dir is not None:
        save_path = Path(save_dir) / f"{model_name.lower()}_{dataset_name}.pkl"
        joblib.dump(model, save_path)
        print(f"Model saved to: {save_path}")
    
    results = {
        'model': model,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'hyperparameters': {
            'init_method': init_method,
            'optimizer': optimizer,
            'lr': lr
        }
    }
    
    return model, results


def train_neural_mtlr(X_train, T_train, E_train, X_val, T_val, E_val,
                      dataset_name, model_name="NeuralMTLR",
                      structure=None, init_method='glorot_uniform',
                      optimizer='adam', lr=8e-4,
                      save_dir=None):
    """
    Train Neural Multi-Task Logistic Regression model
    
    Args:
        X_train: Training features (n_samples, n_features)
        T_train: Training time-to-event (n_samples,)
        E_train: Training event indicator (n_samples,)
        X_val: Validation features
        T_val: Validation time-to-event
        E_val: Validation event indicator
        dataset_name: Name of dataset ('3year' or '10year')
        model_name: Name for the model
        structure: List of hidden layer sizes (e.g., [50, 25])
        init_method: Weight initialization method
        optimizer: Optimizer ('adam', 'sgd', etc.)
        lr: Learning rate
        save_dir: Directory to save model
        
    Returns:
        model: Trained Neural MTLR model
        results: Dictionary with model information
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Default structure if not provided
    if structure is None:
        n_features = X_train.shape[1]
        structure = [min(50, n_features), min(25, n_features // 2)]
    
    print(f"Network structure: {structure}")
    
    # Initialize model
    model = NeuralMultiTaskModel(structure=structure)
    
    # Train model
    print(f"Training with optimizer={optimizer}, lr={lr}...")
    try:
        model.fit(
            X=X_train,
            T=T_train,
            E=E_train,
            init_method=init_method,
            optimizer=optimizer,
            lr=lr
        )
        print("Training completed!")
    except Exception as e:
        print(f"Error training Neural MTLR: {e}")
        print("This may be due to computational constraints. Skipping...")
        return None, None
    
    # Save model
    if save_dir is not None:
        save_path = Path(save_dir) / f"{model_name.lower()}_{dataset_name}.pkl"
        joblib.dump(model, save_path)
        print(f"Model saved to: {save_path}")
    
    results = {
        'model': model,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'structure': structure,
        'hyperparameters': {
            'init_method': init_method,
            'optimizer': optimizer,
            'lr': lr
        }
    }
    
    return model, results
