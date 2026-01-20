"""
Exploratory Data Analysis utilities for clinical deterioration prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_target_distribution(df: pd.DataFrame, title: str = "Target Distribution", 
                            save_path: Optional[str] = None):
    """
    Plot the distribution of the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with target column
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    if 'target' not in df.columns:
        print("No 'target' column found in dataset")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    target_counts = df['target'].value_counts().sort_index()
    axes[0].bar(target_counts.index, target_counts.values, alpha=0.7, color=['skyblue', 'salmon'])
    axes[0].set_xlabel('Target Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{title} - Count')
    axes[0].set_xticks([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    total = len(df)
    for i, (idx, val) in enumerate(target_counts.items()):
        pct = val / total * 100
        axes[0].text(idx, val, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(target_counts.values, labels=[f'Class {i}' for i in target_counts.index], 
                autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    axes[1].set_title(f'{title} - Proportion')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, features: List[str], 
                               target_col: str = 'target', n_cols: int = 3,
                               save_path: Optional[str] = None):
    """
    Plot distributions of features by target class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    features : List[str]
        List of feature names to plot
    target_col : str
        Name of target column
    n_cols : int
        Number of columns in subplot grid
    save_path : str, optional
        Path to save the figure
    """
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        if feature not in df.columns:
            continue
            
        ax = axes[idx]
        
        # Remove missing values for this feature
        plot_df = df[[feature, target_col]].dropna()
        
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature)
            continue
        
        # Plot distributions by target class
        for target_val in sorted(plot_df[target_col].unique()):
            if pd.isna(target_val):
                continue
            subset = plot_df[plot_df[target_col] == target_val][feature]
            if len(subset) > 0:
                ax.hist(subset, alpha=0.6, label=f'Class {int(target_val)}', bins=30)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(feature)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_missing_values(df: pd.DataFrame, title: str = "Missing Values", 
                       save_path: Optional[str] = None):
    """
    Visualize missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    
    if len(missing) == 0:
        print("No missing values found in the dataset")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(missing) * 0.3)))
    
    # Count plot
    axes[0].barh(range(len(missing)), missing.values, alpha=0.7)
    axes[0].set_yticks(range(len(missing)))
    axes[0].set_yticklabels(missing.index)
    axes[0].set_xlabel('Missing Count')
    axes[0].set_title(f'{title} - Count')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Percentage plot
    axes[1].barh(range(len(missing_pct)), missing_pct.values, alpha=0.7, color='coral')
    axes[1].set_yticks(range(len(missing_pct)))
    axes[1].set_yticklabels(missing_pct.index)
    axes[1].set_xlabel('Missing Percentage (%)')
    axes[1].set_title(f'{title} - Percentage')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_statistics_by_target(df: pd.DataFrame, features: List[str], 
                                  target_col: str = 'target') -> pd.DataFrame:
    """
    Calculate descriptive statistics for features grouped by target class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    features : List[str]
        List of feature names
    target_col : str
        Name of target column
        
    Returns:
    --------
    pd.DataFrame
        Statistics summary
    """
    stats_list = []
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        for target_val in sorted(df[target_col].dropna().unique()):
            subset = df[df[target_col] == target_val][feature].dropna()
            
            if len(subset) == 0:
                continue
            
            stats_dict = {
                'feature': feature,
                'target_class': target_val,
                'count': len(subset),
                'mean': subset.mean(),
                'std': subset.std(),
                'median': subset.median(),
                'min': subset.min(),
                'max': subset.max(),
                'q25': subset.quantile(0.25),
                'q75': subset.quantile(0.75),
            }
            stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)


def detect_outliers_iqr(df: pd.DataFrame, features: List[str]) -> dict:
    """
    Detect outliers using IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    features : List[str]
        List of numeric feature names
        
    Returns:
    --------
    dict
        Dictionary with outlier information for each feature
    """
    outliers = {}
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        data = df[feature].dropna()
        if len(data) == 0:
            continue
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_pct = outlier_count / len(data) * 100
        
        outliers[feature] = {
            'count': outlier_count,
            'percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers


def plot_correlation_matrix(df: pd.DataFrame, features: List[str], 
                            target_col: str = 'target', save_path: Optional[str] = None):
    """
    Plot correlation matrix for features and target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    features : List[str]
        List of feature names
    target_col : str
        Name of target column
    save_path : str, optional
        Path to save the figure
    """
    # Select numeric features
    numeric_features = [f for f in features if f in df.columns and df[f].dtype in ['int64', 'float64']]
    
    if target_col in df.columns:
        numeric_features.append(target_col)
    
    corr_df = df[numeric_features].corr()
    
    plt.figure(figsize=(max(10, len(numeric_features) * 0.5), max(8, len(numeric_features) * 0.5)))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_temporal_patterns(df: pd.DataFrame, time_col: str = 'measurement_time',
                            target_col: str = 'target', save_path: Optional[str] = None):
    """
    Analyze temporal patterns in the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    time_col : str
        Name of time column
    target_col : str
        Name of target column
    save_path : str, optional
        Path to save the figure
    """
    if time_col not in df.columns:
        print(f"Time column '{time_col}' not found")
        return
    
    # Create a copy and set time as index
    plot_df = df[[time_col, target_col]].copy()
    plot_df = plot_df.dropna(subset=[time_col])
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[time_col])
    
    if len(plot_df) == 0:
        print("No valid temporal data found")
        return
    
    plot_df = plot_df.set_index(time_col)
    plot_df = plot_df.sort_index()
    
    # Resample by day and count events
    daily_counts = plot_df.groupby([plot_df.index.date, target_col]).size().unstack(fill_value=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series of target distribution
    axes[0].plot(daily_counts.index, daily_counts.get(0, pd.Series(0, index=daily_counts.index)), 
                 label='Class 0', alpha=0.7)
    axes[0].plot(daily_counts.index, daily_counts.get(1, pd.Series(0, index=daily_counts.index)), 
                 label='Class 1', alpha=0.7)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Daily Target Distribution Over Time')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Monthly aggregation
    plot_df['month'] = plot_df.index.to_period('M')
    monthly_counts = plot_df.groupby(['month', target_col]).size().unstack(fill_value=0)
    monthly_counts.index = monthly_counts.index.to_timestamp()
    
    axes[1].bar(monthly_counts.index, monthly_counts.get(0, pd.Series(0, index=monthly_counts.index)), 
                label='Class 0', alpha=0.7, width=20)
    axes[1].bar(monthly_counts.index, monthly_counts.get(1, pd.Series(0, index=monthly_counts.index)), 
                bottom=monthly_counts.get(0, pd.Series(0, index=monthly_counts.index)),
                label='Class 1', alpha=0.7, width=20)
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Monthly Target Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
