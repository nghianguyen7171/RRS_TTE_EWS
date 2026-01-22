"""
Visualization functions for survival analysis models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_survival_curves(model, X, T, E, dataset_name, model_name, save_dir=None):
    """
    Plot survival probability curves
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix
        T: Time-to-event array
        E: Event indicator array
        dataset_name: Dataset name
        model_name: Model name
        save_dir: Directory to save figures
    """
    try:
        # Get time points for prediction
        time_points = np.linspace(0, T.max(), 100)
        
        # Predict survival for all samples
        survival_predictions = []
        for t in time_points:
            try:
                pred = model.predict_survival(X, t=t)
                survival_predictions.append(pred)
            except:
                # If prediction fails, use mean
                survival_predictions.append(np.ones(len(X)) * 0.5)
        
        survival_predictions = np.array(survival_predictions)
        mean_survival = survival_predictions.mean(axis=1)
        std_survival = survival_predictions.std(axis=1)
        
        # Plot overall survival curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_points, mean_survival, linewidth=2, label='Predicted Survival')
        ax.fill_between(time_points, 
                        mean_survival - std_survival, 
                        mean_survival + std_survival, 
                        alpha=0.3)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(f'{model_name} - Overall Survival Curve ({dataset_name})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"survival_curves" / f"{model_name.lower()}_overall_{dataset_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
        # Plot survival by risk quartiles
        try:
            risk_scores = model.predict_risk(X)
            quartiles = np.percentile(risk_scores, [25, 50, 75])
            groups = np.digitize(risk_scores, quartiles)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for q in range(4):
                mask = (groups == q)
                if mask.sum() == 0:
                    continue
                
                group_survival = survival_predictions[:, mask].mean(axis=1)
                ax.plot(time_points, group_survival, linewidth=2, 
                       label=f'Risk Quartile {q+1}')
            
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Survival Probability', fontsize=12)
            ax.set_title(f'{model_name} - Survival by Risk Quartiles ({dataset_name})', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir) / f"survival_curves" / f"{model_name.lower()}_by_risk_{dataset_name}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
            plt.close()
        except:
            pass
        
        # Plot survival by event status (validation)
        fig, ax = plt.subplots(figsize=(10, 6))
        for event_status in [0, 1]:
            mask = (E == event_status)
            if mask.sum() == 0:
                continue
            
            group_survival = survival_predictions[:, mask].mean(axis=1)
            label = 'Censored' if event_status == 0 else 'Event'
            ax.plot(time_points, group_survival, linewidth=2, label=label)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(f'{model_name} - Survival by Event Status ({dataset_name})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"survival_curves" / f"{model_name.lower()}_by_event_{dataset_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting survival curves: {e}")
        import traceback
        traceback.print_exc()


def plot_hazard_curves(model, X, T, E, dataset_name, model_name, save_dir=None):
    """
    Plot hazard function curves
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix
        T: Time-to-event array
        E: Event indicator array
        dataset_name: Dataset name
        model_name: Model name
        save_dir: Directory to save figures
    """
    try:
        # Get time points
        time_points = np.linspace(0, T.max(), 100)
        
        # Predict hazard for all samples
        hazard_predictions = []
        for t in time_points:
            try:
                pred = model.predict_hazard(X, t=t)
                hazard_predictions.append(pred)
            except:
                # If prediction fails, use zeros
                hazard_predictions.append(np.zeros(len(X)))
        
        hazard_predictions = np.array(hazard_predictions)
        mean_hazard = hazard_predictions.mean(axis=1)
        
        # Plot baseline hazard
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_points, mean_hazard, linewidth=2, label='Mean Hazard')
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Hazard Rate', fontsize=12)
        ax.set_title(f'{model_name} - Baseline Hazard Function ({dataset_name})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"hazard_curves" / f"{model_name.lower()}_baseline_{dataset_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
        # Plot hazard by risk groups
        try:
            risk_scores = model.predict_risk(X)
            quartiles = np.percentile(risk_scores, [25, 50, 75])
            groups = np.digitize(risk_scores, quartiles)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for q in range(4):
                mask = (groups == q)
                if mask.sum() == 0:
                    continue
                
                group_hazard = hazard_predictions[:, mask].mean(axis=1)
                ax.plot(time_points, group_hazard, linewidth=2, 
                       label=f'Risk Quartile {q+1}')
            
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Hazard Rate', fontsize=12)
            ax.set_title(f'{model_name} - Hazard by Risk Groups ({dataset_name})', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir) / f"hazard_curves" / f"{model_name.lower()}_by_risk_{dataset_name}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
            plt.close()
        except:
            pass
        
    except Exception as e:
        print(f"Error plotting hazard curves: {e}")


def plot_risk_distributions(model, X, T, E, dataset_name, model_name, save_dir=None):
    """
    Plot risk score distributions by event status
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix
        T: Time-to-event array
        E: Event indicator array
        dataset_name: Dataset name
        model_name: Model name
        save_dir: Directory to save figures
    """
    try:
        # Get risk scores
        try:
            risk_scores = model.predict_risk(X)
        except:
            # Fallback
            median_time = np.median(T[E == 1]) if (E == 1).sum() > 0 else np.median(T)
            survival_probs = model.predict_survival(X, t=median_time)
            risk_scores = -np.log(survival_probs + 1e-10)
        
        # Histogram by event status
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(risk_scores[E == 0], bins=50, alpha=0.6, label='Censored', density=True)
        ax.hist(risk_scores[E == 1], bins=50, alpha=0.6, label='Event', density=True)
        ax.set_xlabel('Risk Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{model_name} - Risk Score Distribution ({dataset_name})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"risk_distributions" / f"{model_name.lower()}_histogram_{dataset_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
        # Boxplot by risk quartiles
        try:
            quartiles = np.percentile(risk_scores, [25, 50, 75])
            groups = np.digitize(risk_scores, quartiles)
            
            data_for_box = []
            labels = []
            for q in range(4):
                mask = (groups == q)
                if mask.sum() > 0:
                    data_for_box.append(risk_scores[mask])
                    labels.append(f'Q{q+1}')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data_for_box, labels=labels)
            ax.set_xlabel('Risk Quartile', fontsize=12)
            ax.set_ylabel('Risk Score', fontsize=12)
            ax.set_title(f'{model_name} - Risk Scores by Quartiles ({dataset_name})', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir) / f"risk_distributions" / f"{model_name.lower()}_boxplot_{dataset_name}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
            plt.close()
        except:
            pass
        
    except Exception as e:
        print(f"Error plotting risk distributions: {e}")


def plot_calibration_curves(calibration_df, dataset_name, model_name, save_dir=None):
    """
    Plot calibration curves (observed vs predicted survival)
    
    Args:
        calibration_df: DataFrame with calibration results
        dataset_name: Dataset name
        model_name: Model name
        save_dir: Directory to save figures
    """
    try:
        if calibration_df is None or len(calibration_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(calibration_df['predicted_survival'], 
                  calibration_df['observed_survival'],
                  s=calibration_df['n_samples'] * 10,
                  alpha=0.6)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Predicted Survival Probability', fontsize=12)
        ax.set_ylabel('Observed Survival Probability', fontsize=12)
        ax.set_title(f'{model_name} - Calibration Plot ({dataset_name})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"risk_distributions" / f"{model_name.lower()}_calibration_{dataset_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting calibration curves: {e}")


def plot_model_comparison(metrics_df, dataset_name, save_dir=None):
    """
    Compare all models (C-index, IBS)
    
    Args:
        metrics_df: DataFrame with model metrics
        dataset_name: Dataset name
        save_dir: Directory to save figures
    """
    try:
        if metrics_df is None or len(metrics_df) == 0:
            return
        
        # C-index comparison
        if 'c_index_test' in metrics_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            models = metrics_df['model'].values
            c_indices = metrics_df['c_index_test'].values
            
            bars = ax.barh(models, c_indices)
            ax.set_xlabel('C-index (Concordance)', fontsize=12)
            ax.set_title(f'Model Comparison - C-index ({dataset_name})', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, c_indices)):
                if not np.isnan(val):
                    ax.text(val + 0.01, i, f'{val:.3f}', va='center')
            
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir) / f"model_comparison" / f"c_index_comparison_{dataset_name}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
            plt.close()
        
        # IBS comparison
        if 'ibs_test' in metrics_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            models = metrics_df['model'].values
            ibs_values = metrics_df['ibs_test'].values
            
            bars = ax.barh(models, ibs_values)
            ax.set_xlabel('Integrated Brier Score (lower is better)', fontsize=12)
            ax.set_title(f'Model Comparison - Integrated Brier Score ({dataset_name})', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, ibs_values)):
                if not np.isnan(val):
                    ax.text(val + 0.01, i, f'{val:.3f}', va='center')
            
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir) / f"model_comparison" / f"ibs_comparison_{dataset_name}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {save_path}")
            plt.close()
        
    except Exception as e:
        print(f"Error plotting model comparison: {e}")


def plot_risk_stratification(model, X, T, E, dataset_name, model_name, save_dir=None):
    """
    Plot Kaplan-Meier curves by risk quartiles
    
    Args:
        model: Trained PySurvival model
        X: Feature matrix
        T: Time-to-event array
        E: Event indicator array
        dataset_name: Dataset name
        model_name: Model name
        save_dir: Directory to save figures
    """
    try:
        # Get risk scores
        try:
            risk_scores = model.predict_risk(X)
        except:
            median_time = np.median(T[E == 1]) if (E == 1).sum() > 0 else np.median(T)
            survival_probs = model.predict_survival(X, t=median_time)
            risk_scores = -np.log(survival_probs + 1e-10)
        
        # Create risk quartiles
        quartiles = np.percentile(risk_scores, [25, 50, 75])
        groups = np.digitize(risk_scores, quartiles)
        
        # Plot KM curves for each quartile
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for q in range(4):
            mask = (groups == q)
            if mask.sum() == 0:
                continue
            
            kmf = KaplanMeierFitter()
            kmf.fit(T[mask], E[mask], label=f'Risk Quartile {q+1}')
            kmf.plot_survival_function(ax=ax, linewidth=2)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(f'{model_name} - KM Curves by Risk Quartiles ({dataset_name})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"survival_curves" / f"{model_name.lower()}_km_quartiles_{dataset_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting risk stratification: {e}")
