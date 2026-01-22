"""
Cox Proportional Hazards Model using PySurvival
"""

import numpy as np
import pandas as pd
from pysurvival.models.semi_parametric import CoxPHModel
import joblib
from pathlib import Path


def train_cox_ph(X_train, T_train, E_train, X_val, T_val, E_val, 
                 dataset_name, model_name="CoxPH", 
                 l2_reg=1e-4, lr=0.4, tol=1e-4, 
                 save_dir=None):
    """
    Train Cox Proportional Hazards model
    
    Args:
        X_train: Training features (n_samples, n_features)
        T_train: Training time-to-event (n_samples,)
        E_train: Training event indicator (n_samples,)
        X_val: Validation features
        T_val: Validation time-to-event
        E_val: Validation event indicator
        dataset_name: Name of dataset ('3year' or '10year')
        model_name: Name for the model
        l2_reg: L2 regularization parameter
        lr: Learning rate
        tol: Convergence tolerance
        save_dir: Directory to save model
        
    Returns:
        model: Trained Cox PH model
        results: Dictionary with model information
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Initialize model
    model = CoxPHModel()
    
    # Train model
    print(f"Training with l2_reg={l2_reg}, lr={lr}, tol={tol}...")
    model.fit(
        X=X_train,
        T=T_train,
        E=E_train,
        init_method='he_uniform',
        l2_reg=l2_reg,
        lr=0.1,  # Reduced learning rate to avoid gradient explosion
        tol=tol
    )
    
    print("Training completed!")
    
    # Get coefficients
    try:
        coefficients = model.coef_
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'hazard_ratio': np.exp(coefficients)
        })
        coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)
        
        print(f"\nTop 10 features by |coefficient|:")
        print(coef_df.head(10).to_string(index=False))
    except:
        coef_df = None
    
    # Save model
    if save_dir is not None:
        save_path = Path(save_dir) / f"{model_name.lower()}_{dataset_name}.pkl"
        joblib.dump(model, save_path)
        print(f"Model saved to: {save_path}")
        
        if coef_df is not None:
            coef_path = Path(save_dir).parent / "metrics" / f"{model_name.lower()}_coefficients_{dataset_name}.csv"
            coef_path.parent.mkdir(parents=True, exist_ok=True)
            coef_df.to_csv(coef_path, index=False)
            print(f"Coefficients saved to: {coef_path}")
    
    results = {
        'model': model,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'coefficients': coef_df,
        'hyperparameters': {
            'l2_reg': l2_reg,
            'lr': lr,
            'tol': tol
        }
    }
    
    return model, results
