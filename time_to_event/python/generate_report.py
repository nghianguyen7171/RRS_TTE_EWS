"""
Generate comprehensive markdown report for survival analysis models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_report(dataset_name, metrics_dir, figures_dir, output_path=None):
    """
    Generate comprehensive markdown report
    
    Args:
        dataset_name: Dataset name ('3year' or '10year')
        metrics_dir: Directory containing metrics CSV files
        figures_dir: Directory containing figure files
        output_path: Path to save report (if None, saves to results/python_baseline/)
    """
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    
    if output_path is None:
        output_path = metrics_dir.parent / f"PYTHON_BASELINE_REPORT_{dataset_name}.md"
    else:
        output_path = Path(output_path)
    
    # Load metrics
    metrics_file = metrics_dir / f"all_metrics_{dataset_name}.csv"
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file)
    else:
        metrics_df = pd.DataFrame()
        print(f"Warning: Metrics file not found: {metrics_file}")
    
    # Start building report
    report_lines = []
    
    # Header
    report_lines.append("# Python Baseline Survival Analysis Report")
    report_lines.append(f"## {dataset_name.upper()} Dataset")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Analysis Framework:** PySurvival")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report presents a comprehensive evaluation of traditional survival analysis models")
    report_lines.append("implemented using the PySurvival library. All models were trained with patient-level")
    report_lines.append("train/validation/test splitting to avoid data leakage.")
    report_lines.append("")
    
    if len(metrics_df) > 0:
        # Filter out NaN values for best model selection
        valid_metrics = metrics_df[metrics_df['c_index_test'].notna()]
        if len(valid_metrics) > 0:
            best_model = valid_metrics.loc[valid_metrics['c_index_test'].idxmax()]
            report_lines.append(f"**Best Model (by C-index):** {best_model['model']} (C-index: {best_model['c_index_test']:.3f})")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Model Performance Comparison
    report_lines.append("## Model Performance Comparison")
    report_lines.append("")
    
    if len(metrics_df) > 0:
        report_lines.append("### Performance Metrics Table")
        report_lines.append("")
        report_lines.append("| Model | C-index (Train) | C-index (Val) | C-index (Test) | IBS (Val) | IBS (Test) | ECE (Val) | ECE (Test) |")
        report_lines.append("|-------|----------------|---------------|----------------|-----------|------------|-----------|------------|")
        
        for _, row in metrics_df.iterrows():
            c_train = f"{row['c_index_train']:.3f}" if pd.notna(row['c_index_train']) else "N/A"
            c_val = f"{row['c_index_val']:.3f}" if pd.notna(row['c_index_val']) else "N/A"
            c_test = f"{row['c_index_test']:.3f}" if pd.notna(row['c_index_test']) else "N/A"
            ibs_val = f"{row['ibs_val']:.3f}" if pd.notna(row['ibs_val']) else "N/A"
            ibs_test = f"{row['ibs_test']:.3f}" if pd.notna(row['ibs_test']) else "N/A"
            ece_val = f"{row['ece_val']:.3f}" if pd.notna(row['ece_val']) else "N/A"
            ece_test = f"{row['ece_test']:.3f}" if pd.notna(row['ece_test']) else "N/A"
            
            report_lines.append(f"| {row['model']} | {c_train} | {c_val} | {c_test} | {ibs_val} | {ibs_test} | {ece_val} | {ece_test} |")
        
        report_lines.append("")
        report_lines.append("**Legend:**")
        report_lines.append("- **C-index (Concordance):** Measures discrimination ability (0.5 = random, 1.0 = perfect)")
        report_lines.append("- **IBS (Integrated Brier Score):** Measures prediction accuracy (lower is better, 0 = perfect)")
        report_lines.append("- **ECE (Expected Calibration Error):** Measures calibration (lower is better, 0 = perfect)")
        report_lines.append("")
    else:
        report_lines.append("No metrics available.")
        report_lines.append("")
    
    # Model Comparison Figures
    report_lines.append("### Model Comparison Visualizations")
    report_lines.append("")
    
    c_index_fig = figures_dir / "model_comparison" / f"c_index_comparison_{dataset_name}.png"
    ibs_fig = figures_dir / "model_comparison" / f"ibs_comparison_{dataset_name}.png"
    
    if c_index_fig.exists():
        report_lines.append(f"![C-index Comparison]({c_index_fig.relative_to(output_path.parent)})")
        report_lines.append("")
        report_lines.append("*Figure 1: C-index comparison across all models. Higher values indicate better discrimination.*")
        report_lines.append("")
    
    if ibs_fig.exists():
        report_lines.append(f"![IBS Comparison]({ibs_fig.relative_to(output_path.parent)})")
        report_lines.append("")
        report_lines.append("*Figure 2: Integrated Brier Score comparison. Lower values indicate better prediction accuracy.*")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Individual Model Results
    report_lines.append("## Individual Model Results")
    report_lines.append("")
    
    if len(metrics_df) > 0:
        for _, row in metrics_df.iterrows():
            model_name = row['model']
            report_lines.append(f"### {model_name}")
            report_lines.append("")
            
            # Performance summary
            report_lines.append(f"**Performance Summary:**")
            report_lines.append(f"- C-index (Test): {row['c_index_test']:.3f}" if pd.notna(row['c_index_test']) else "- C-index (Test): N/A")
            report_lines.append(f"- IBS (Test): {row['ibs_test']:.3f}" if pd.notna(row['ibs_test']) else "- IBS (Test): N/A")
            report_lines.append(f"- ECE (Test): {row['ece_test']:.3f}" if pd.notna(row['ece_test']) else "- ECE (Test): N/A")
            report_lines.append("")
            
            # Figures
            model_name_lower = model_name.lower()
            
            # Survival curves
            survival_overall = figures_dir / "survival_curves" / f"{model_name_lower}_overall_{dataset_name}.png"
            survival_risk = figures_dir / "survival_curves" / f"{model_name_lower}_by_risk_{dataset_name}.png"
            survival_event = figures_dir / "survival_curves" / f"{model_name_lower}_by_event_{dataset_name}.png"
            
            if survival_overall.exists():
                report_lines.append(f"![Overall Survival Curve]({survival_overall.relative_to(output_path.parent)})")
                report_lines.append("")
                report_lines.append(f"*Figure: Overall survival curve for {model_name}.*")
                report_lines.append("")
            
            if survival_risk.exists():
                report_lines.append(f"![Survival by Risk Quartiles]({survival_risk.relative_to(output_path.parent)})")
                report_lines.append("")
                report_lines.append(f"*Figure: Survival curves stratified by risk quartiles for {model_name}.*")
                report_lines.append("")
            
            if survival_event.exists():
                report_lines.append(f"![Survival by Event Status]({survival_event.relative_to(output_path.parent)})")
                report_lines.append("")
                report_lines.append(f"*Figure: Survival curves by event status (validation) for {model_name}.*")
                report_lines.append("")
            
            # Risk distributions
            risk_hist = figures_dir / "risk_distributions" / f"{model_name_lower}_histogram_{dataset_name}.png"
            risk_box = figures_dir / "risk_distributions" / f"{model_name_lower}_boxplot_{dataset_name}.png"
            
            if risk_hist.exists():
                report_lines.append(f"![Risk Score Distribution]({risk_hist.relative_to(output_path.parent)})")
                report_lines.append("")
                report_lines.append(f"*Figure: Risk score distribution by event status for {model_name}.*")
                report_lines.append("")
            
            if risk_box.exists():
                report_lines.append(f"![Risk Scores by Quartiles]({risk_box.relative_to(output_path.parent)})")
                report_lines.append("")
                report_lines.append(f"*Figure: Risk score boxplots by quartiles for {model_name}.*")
                report_lines.append("")
            
            # Hazard curves
            hazard_baseline = figures_dir / "hazard_curves" / f"{model_name_lower}_baseline_{dataset_name}.png"
            hazard_risk = figures_dir / "hazard_curves" / f"{model_name_lower}_by_risk_{dataset_name}.png"
            
            if hazard_baseline.exists():
                report_lines.append(f"![Baseline Hazard]({hazard_baseline.relative_to(output_path.parent)})")
                report_lines.append("")
                report_lines.append(f"*Figure: Baseline hazard function for {model_name}.*")
                report_lines.append("")
            
            if hazard_risk.exists():
                report_lines.append(f"![Hazard by Risk Groups]({hazard_risk.relative_to(output_path.parent)})")
                report_lines.append("")
                report_lines.append(f"*Figure: Hazard functions by risk groups for {model_name}.*")
                report_lines.append("")
            
            report_lines.append("---")
            report_lines.append("")
    
    # Model Descriptions
    report_lines.append("## Model Descriptions")
    report_lines.append("")
    
    model_descriptions = {
        'CoxPH': """
**Cox Proportional Hazards Model**
- Semi-parametric model that assumes proportional hazards
- Estimates hazard ratios for each feature
- Does not require specification of baseline hazard
- Most commonly used survival model in medical research
        """,
        'MTLR': """
**Multi-Task Logistic Regression (Linear)**
- Parametric model that predicts survival probabilities at multiple time points
- Uses logistic regression at each time point
- Can handle non-proportional hazards
        """,
        'NeuralMTLR': """
**Neural Multi-Task Logistic Regression**
- Neural network extension of MTLR
- Can capture non-linear relationships
- More flexible but requires more data
        """,
        'RandomSurvivalForest': """
**Random Survival Forest**
- Tree-based ensemble method
- Handles non-linear relationships and interactions
- Provides feature importance
- Robust to outliers
        """,
        'ExtraSurvivalTrees': """
**Extra Survival Trees**
- Variant of Random Survival Forest with more randomness
- Faster training than Random Survival Forest
- Good for high-dimensional data
        """,
        'ConditionalSurvivalForest': """
**Conditional Survival Forest**
- Uses conditional inference trees
- Statistically principled splitting
- Good for small sample sizes
        """,
        'LinearSurvivalSVM': """
**Linear Survival Support Vector Machine**
- Linear kernel SVM for survival analysis
- Good for high-dimensional data
- Fast training
        """,
        'KernelSurvivalSVM': """
**Kernel Survival Support Vector Machine**
- Non-linear SVM using RBF or polynomial kernels
- Can capture complex relationships
- Slower than linear SVM
        """
    }
    
    for model_name, description in model_descriptions.items():
        if model_name in metrics_df['model'].values if len(metrics_df) > 0 else False:
            report_lines.append(f"### {model_name}")
            report_lines.append(description.strip())
            report_lines.append("")
    
    # Conclusions
    report_lines.append("## Conclusions")
    report_lines.append("")
    
    if len(metrics_df) > 0:
        valid_metrics = metrics_df[metrics_df['c_index_test'].notna()]
        if len(valid_metrics) > 0:
            best_model = valid_metrics.loc[valid_metrics['c_index_test'].idxmax()]
            report_lines.append(f"1. **Best Performing Model:** {best_model['model']} achieved the highest C-index of {best_model['c_index_test']:.3f} on the test set.")
        else:
            report_lines.append("1. **Model Performance:** Training completed but metrics need review.")
        report_lines.append("")
        report_lines.append("2. **Model Comparison:** All models were evaluated using patient-level splitting to ensure")
        report_lines.append("   realistic performance estimates and avoid data leakage.")
        report_lines.append("")
        report_lines.append("3. **Performance Metrics:** Models were evaluated using:")
        report_lines.append("   - C-index (concordance) for discrimination")
        report_lines.append("   - Integrated Brier Score for prediction accuracy")
        report_lines.append("   - Expected Calibration Error for calibration")
        report_lines.append("")
        report_lines.append("4. **Recommendations:**")
        report_lines.append("   - The best model should be validated on external datasets")
        report_lines.append("   - Consider ensemble methods combining top-performing models")
        report_lines.append("   - Further hyperparameter tuning may improve performance")
        report_lines.append("")
    else:
        report_lines.append("No models were successfully trained. Please check the training logs for errors.")
        report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text)
    
    print(f"Report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate markdown report')
    parser.add_argument('--dataset', type=str, required=True, choices=['3year', '10year'],
                       help='Dataset name')
    parser.add_argument('--metrics-dir', type=str, default=None,
                       help='Directory containing metrics CSV files')
    parser.add_argument('--figures-dir', type=str, default=None,
                       help='Directory containing figure files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for report')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.metrics_dir is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        args.metrics_dir = project_root / "results" / "python_baseline" / "metrics"
    
    if args.figures_dir is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        args.figures_dir = project_root / "results" / "python_baseline" / "figures"
    
    generate_report(args.dataset, args.metrics_dir, args.figures_dir, args.output)
