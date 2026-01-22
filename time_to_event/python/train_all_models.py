"""
Main script to train all survival analysis models
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from python.data_loader import load_survival_data, prepare_pysurvival_format
from python.patient_splitter import patient_level_split
from python.evaluation import calculate_all_metrics
from python.visualization import (
    plot_survival_curves, plot_hazard_curves, plot_risk_distributions,
    plot_calibration_curves, plot_model_comparison, plot_risk_stratification
)

# Import model training functions
from python.models.cox_ph import train_cox_ph
from python.models.mtlr import train_mtlr, train_neural_mtlr
from python.models.parametric import train_parametric_models
from python.models.survival_forest import train_survival_forests
try:
    from python.models.survival_svm import train_survival_svm
except ImportError:
    train_survival_svm = None


def train_all_models(dataset_name, models_to_train=None, save_dir=None):
    """
    Train all survival analysis models for a dataset
    
    Args:
        dataset_name: '3year' or '10year'
        models_to_train: List of model types to train (if None, trains all)
        save_dir: Base directory to save results
    """
    print(f"\n{'='*80}")
    print(f"TRAINING ALL MODELS FOR {dataset_name.upper()} DATASET")
    print(f"{'='*80}\n")
    
    # Set up directories
    if save_dir is None:
        save_dir = project_root / "results" / "python_baseline"
    
    save_dir = Path(save_dir)
    models_dir = save_dir / "models"
    figures_dir = save_dir / "figures"
    metrics_dir = save_dir / "metrics"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X, T, E, metadata = load_survival_data(dataset_name)
    
    # Prepare PySurvival format
    X_array, T_array, E_array = prepare_pysurvival_format(X, T, E)
    
    # Patient-level split
    print("\nPerforming patient-level split...")
    splits = patient_level_split(X, T_array, E_array, metadata,
                                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Prepare splits
    X_train = splits['X_train'].values if isinstance(splits['X_train'], pd.DataFrame) else splits['X_train']
    T_train = splits['T_train']
    E_train = splits['E_train']
    
    X_val = splits['X_val'].values if isinstance(splits['X_val'], pd.DataFrame) else splits['X_val']
    T_val = splits['T_val']
    E_val = splits['E_val']
    
    X_test = splits['X_test'].values if isinstance(splits['X_test'], pd.DataFrame) else splits['X_test']
    T_test = splits['T_test']
    E_test = splits['E_test']
    
    # Store all results
    all_models = {}
    all_metrics = []
    
    # Define models to train
    if models_to_train is None:
        models_to_train = [
            'CoxPH',
            'MTLR',
            'NeuralMTLR',
            'Parametric',
            'RandomSurvivalForest',
            'ExtraSurvivalTrees',
            'ConditionalSurvivalForest',
            'LinearSurvivalSVM',
            'KernelSurvivalSVM'
        ]
    
    # 1. Train Cox PH
    if 'CoxPH' in models_to_train:
        try:
            model, results = train_cox_ph(
                X_train, T_train, E_train, X_val, T_val, E_val,
                dataset_name, model_name="CoxPH",
                save_dir=models_dir
            )
            all_models['CoxPH'] = model
            
            # Evaluate
            metrics = calculate_all_metrics(
                model, X_train, T_train, E_train,
                X_val, T_val, E_val, X_test, T_test, E_test,
                dataset_name, "CoxPH"
            )
            all_metrics.append(metrics)
            
            # Visualize
            plot_survival_curves(model, X_test, T_test, E_test, dataset_name, "CoxPH", figures_dir)
            plot_hazard_curves(model, X_test, T_test, E_test, dataset_name, "CoxPH", figures_dir)
            plot_risk_distributions(model, X_test, T_test, E_test, dataset_name, "CoxPH", figures_dir)
            plot_risk_stratification(model, X_test, T_test, E_test, dataset_name, "CoxPH", figures_dir)
            
            if metrics.get('stratification_val') is not None:
                plot_calibration_curves(metrics['stratification_val'], dataset_name, "CoxPH", figures_dir)
        except Exception as e:
            print(f"Error training CoxPH: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. Train MTLR
    if 'MTLR' in models_to_train:
        try:
            model, results = train_mtlr(
                X_train, T_train, E_train, X_val, T_val, E_val,
                dataset_name, model_name="MTLR",
                save_dir=models_dir
            )
            if model is not None:
                all_models['MTLR'] = model
                
                metrics = calculate_all_metrics(
                    model, X_train, T_train, E_train,
                    X_val, T_val, E_val, X_test, T_test, E_test,
                    dataset_name, "MTLR"
                )
                all_metrics.append(metrics)
                
                plot_survival_curves(model, X_test, T_test, E_test, dataset_name, "MTLR", figures_dir)
                plot_risk_distributions(model, X_test, T_test, E_test, dataset_name, "MTLR", figures_dir)
        except Exception as e:
            print(f"Error training MTLR: {e}")
    
    # 3. Train Neural MTLR (optional, may be slow)
    if 'NeuralMTLR' in models_to_train:
        try:
            model, results = train_neural_mtlr(
                X_train, T_train, E_train, X_val, T_val, E_val,
                dataset_name, model_name="NeuralMTLR",
                save_dir=models_dir
            )
            if model is not None:
                all_models['NeuralMTLR'] = model
                
                metrics = calculate_all_metrics(
                    model, X_train, T_train, E_train,
                    X_val, T_val, E_val, X_test, T_test, E_test,
                    dataset_name, "NeuralMTLR"
                )
                all_metrics.append(metrics)
                
                plot_survival_curves(model, X_test, T_test, E_test, dataset_name, "NeuralMTLR", figures_dir)
                plot_risk_distributions(model, X_test, T_test, E_test, dataset_name, "NeuralMTLR", figures_dir)
        except Exception as e:
            print(f"Error training NeuralMTLR: {e}")
    
    # 4. Train Parametric Models
    if 'Parametric' in models_to_train:
        try:
            models_dict, comparison_df = train_parametric_models(
                X_train, T_train, E_train, X_val, T_val, E_val,
                dataset_name, save_dir=models_dir
            )
            
            # Evaluate best parametric model (lowest AIC)
            if len(comparison_df) > 0 and 'AIC' in comparison_df.columns:
                best_model_name = comparison_df.loc[comparison_df['AIC'].idxmin(), 'model']
                if best_model_name in models_dict:
                    model = models_dict[best_model_name]
                    all_models[f'Parametric_{best_model_name}'] = model
                    
                    metrics = calculate_all_metrics(
                        model, X_train, T_train, E_train,
                        X_val, T_val, E_val, X_test, T_test, E_test,
                        dataset_name, f"Parametric_{best_model_name}"
                    )
                    all_metrics.append(metrics)
                    
                    plot_survival_curves(model, X_test, T_test, E_test, dataset_name, 
                                       f"Parametric_{best_model_name}", figures_dir)
        except Exception as e:
            print(f"Error training Parametric models: {e}")
    
    # 5. Train Survival Forests
    if any(x in models_to_train for x in ['RandomSurvivalForest', 'ExtraSurvivalTrees', 'ConditionalSurvivalForest']):
        try:
            forest_models = ['RandomSurvivalForest', 'ExtraSurvivalTrees', 'ConditionalSurvivalForest']
            models_to_train_forests = [x for x in forest_models if x in models_to_train]
            
            models_dict = train_survival_forests(
                X_train, T_train, E_train, X_val, T_val, E_val,
                dataset_name, models_to_train=models_to_train_forests,
                n_estimators=100, save_dir=models_dir
            )
            
            for model_name, model in models_dict.items():
                all_models[model_name] = model
                
                metrics = calculate_all_metrics(
                    model, X_train, T_train, E_train,
                    X_val, T_val, E_val, X_test, T_test, E_test,
                    dataset_name, model_name
                )
                all_metrics.append(metrics)
                
                plot_survival_curves(model, X_test, T_test, E_test, dataset_name, model_name, figures_dir)
                plot_risk_distributions(model, X_test, T_test, E_test, dataset_name, model_name, figures_dir)
        except Exception as e:
            print(f"Error training Survival Forests: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. Train Survival SVM
    if train_survival_svm is not None and any(x in models_to_train for x in ['LinearSurvivalSVM', 'KernelSurvivalSVM']):
        try:
            svm_models = []
            if 'LinearSurvivalSVM' in models_to_train:
                svm_models.append('Linear')
            if 'KernelSurvivalSVM' in models_to_train:
                svm_models.append('Kernel')
            
            models_dict = train_survival_svm(
                X_train, T_train, E_train, X_val, T_val, E_val,
                dataset_name, models_to_train=svm_models,
                save_dir=models_dir
            )
            
            for model_name, model in models_dict.items():
                all_models[model_name] = model
                
                metrics = calculate_all_metrics(
                    model, X_train, T_train, E_train,
                    X_val, T_val, E_val, X_test, T_test, E_test,
                    dataset_name, model_name
                )
                all_metrics.append(metrics)
                
                plot_survival_curves(model, X_test, T_test, E_test, dataset_name, model_name, figures_dir)
                plot_risk_distributions(model, X_test, T_test, E_test, dataset_name, model_name, figures_dir)
        except Exception as e:
            print(f"Error training Survival SVM: {e}")
    
    # Create metrics summary DataFrame
    metrics_summary = []
    for m in all_metrics:
        metrics_summary.append({
            'model': m.get('model', 'Unknown'),
            'dataset': m.get('dataset', dataset_name),
            'c_index_train': m.get('c_index_train'),
            'c_index_val': m.get('c_index_val'),
            'c_index_test': m.get('c_index_test'),
            'ibs_val': m.get('ibs_val'),
            'ibs_test': m.get('ibs_test'),
            'ece_val': m.get('ece_val'),
            'ece_test': m.get('ece_test'),
        })
    
    metrics_df = pd.DataFrame(metrics_summary)
    
    # Save metrics
    metrics_path = metrics_dir / f"all_metrics_{dataset_name}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Plot model comparison
    plot_model_comparison(metrics_df, dataset_name, figures_dir)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE FOR {dataset_name.upper()} DATASET")
    print(f"{'='*80}\n")
    
    return all_models, metrics_df, all_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all survival analysis models')
    parser.add_argument('--dataset', type=str, default='3year', choices=['3year', '10year', 'both'],
                       help='Dataset to train on')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to train (default: all)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    datasets = ['3year', '10year'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_name in datasets:
        train_all_models(dataset_name, models_to_train=args.models, save_dir=args.save_dir)
