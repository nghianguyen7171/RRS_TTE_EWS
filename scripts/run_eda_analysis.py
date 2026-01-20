#!/usr/bin/env python3
"""
Comprehensive EDA Analysis Script
Generates detailed analysis results for both datasets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_3year_dataset, load_10year_dataset, get_feature_groups, validate_dataset
from src.eda_utils import (
    plot_target_distribution,
    plot_feature_distributions,
    plot_missing_values,
    calculate_statistics_by_target,
    detect_outliers_iqr,
    plot_correlation_matrix
)

# Create output directory
output_dir = Path("../reports")
output_dir.mkdir(exist_ok=True)
figures_dir = output_dir / "figures"
figures_dir.mkdir(exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE EDA ANALYSIS")
print("=" * 80)

# ============================================================================
# 3-YEAR DATASET ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("ANALYZING 3-YEAR DATASET")
print("=" * 80)

df_3year = load_3year_dataset("../data/CNUH_3Y.csv")
print(f"Loaded: {df_3year.shape[0]:,} rows × {df_3year.shape[1]} columns")

# Validation
val_3year = validate_dataset(df_3year, "3-year dataset")
print(f"\nMemory usage: {val_3year['memory_usage_mb']:.2f} MB")
print(f"Duplicate rows: {val_3year['duplicate_rows']:,}")

# Target distribution
print("\n" + "-" * 80)
print("TARGET DISTRIBUTION - 3-YEAR DATASET")
print("-" * 80)
target_dist_3year = df_3year['target'].value_counts().sort_index()
for target_val, count in target_dist_3year.items():
    pct = count / len(df_3year) * 100
    print(f"Class {target_val}: {count:,} ({pct:.2f}%)")

# Feature groups
features_3year = get_feature_groups(df_3year)
vital_signs_3year = features_3year['vital_signs']
lab_values_3year = features_3year['lab_values']
demographics_3year = features_3year['demographics']

print(f"\nVital signs ({len(vital_signs_3year)}): {vital_signs_3year}")
print(f"Lab values ({len(lab_values_3year)}): {lab_values_3year}")
print(f"Demographics ({len(demographics_3year)}): {demographics_3year}")

# Missing values
print("\n" + "-" * 80)
print("MISSING VALUES - 3-YEAR DATASET (Top 10)")
print("-" * 80)
missing_3year = df_3year.isnull().sum().sort_values(ascending=False)
for col, count in missing_3year.head(10).items():
    if count > 0:
        pct = count / len(df_3year) * 100
        print(f"{col:30s}: {count:8,} ({pct:6.2f}%)")

# Statistics by target
print("\n" + "-" * 80)
print("STATISTICS BY TARGET - 3-YEAR DATASET")
print("-" * 80)
# Select key features for analysis
key_features_3year = vital_signs_3year[:3] + lab_values_3year[:5] + demographics_3year
key_features_3year = [f for f in key_features_3year if f in df_3year.columns]

stats_3year = calculate_statistics_by_target(df_3year, key_features_3year[:8], 'target')
print("\nSample statistics (first 3 features):")
print(stats_3year.head(6).to_string(index=False))

# Outliers
print("\n" + "-" * 80)
print("OUTLIER DETECTION - 3-YEAR DATASET")
print("-" * 80)
numeric_features_3year = [f for f in key_features_3year if df_3year[f].dtype in ['int64', 'float64']]
outliers_3year = detect_outliers_iqr(df_3year, numeric_features_3year[:5])
for feat, info in list(outliers_3year.items())[:5]:
    print(f"{feat:30s}: {info['count']:6,} outliers ({info['percentage']:5.2f}%)")

# ============================================================================
# 10-YEAR DATASET ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("ANALYZING 10-YEAR DATASET")
print("=" * 80)

df_10year = load_10year_dataset("../data/10yrs_proc.csv")
print(f"Loaded: {df_10year.shape[0]:,} rows × {df_10year.shape[1]} columns")

# Validation
val_10year = validate_dataset(df_10year, "10-year dataset")
print(f"\nMemory usage: {val_10year['memory_usage_mb']:.2f} MB")
print(f"Duplicate rows: {val_10year['duplicate_rows']:,}")

# Target distribution
print("\n" + "-" * 80)
print("TARGET DISTRIBUTION - 10-YEAR DATASET")
print("-" * 80)
target_dist_10year = df_10year['target'].value_counts().sort_index()
for target_val, count in target_dist_10year.items():
    pct = count / len(df_10year) * 100
    print(f"Class {target_val}: {count:,} ({pct:.2f}%)")

# Feature groups
features_10year = get_feature_groups(df_10year)
vital_signs_10year = features_10year['vital_signs']
lab_values_10year = features_10year['lab_values']
demographics_10year = features_10year['demographics']

print(f"\nVital signs ({len(vital_signs_10year)}): {vital_signs_10year}")
print(f"Lab values ({len(lab_values_10year)}): {lab_values_10year}")
print(f"Demographics ({len(demographics_10year)}): {demographics_10year}")

# Missing values
print("\n" + "-" * 80)
print("MISSING VALUES - 10-YEAR DATASET (Top 10)")
print("-" * 80)
missing_10year = df_10year.isnull().sum().sort_values(ascending=False)
for col, count in missing_10year.head(10).items():
    if count > 0:
        pct = count / len(df_10year) * 100
        print(f"{col:30s}: {count:8,} ({pct:6.2f}%)")

# Statistics by target
print("\n" + "-" * 80)
print("STATISTICS BY TARGET - 10-YEAR DATASET")
print("-" * 80)
key_features_10year = vital_signs_10year[:3] + lab_values_10year[:5] + demographics_10year
key_features_10year = [f for f in key_features_10year if f in df_10year.columns]

stats_10year = calculate_statistics_by_target(df_10year, key_features_10year[:8], 'target')
print("\nSample statistics (first 3 features):")
print(stats_10year.head(6).to_string(index=False))

# Outliers
print("\n" + "-" * 80)
print("OUTLIER DETECTION - 10-YEAR DATASET")
print("-" * 80)
numeric_features_10year = [f for f in key_features_10year if df_10year[f].dtype in ['int64', 'float64']]
outliers_10year = detect_outliers_iqr(df_10year, numeric_features_10year[:5])
for feat, info in list(outliers_10year.items())[:5]:
    print(f"{feat:30s}: {info['count']:6,} outliers ({info['percentage']:5.2f}%)")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("DATASET COMPARISON SUMMARY")
print("=" * 80)

print(f"\n{'Metric':<30s} {'3-Year':>15s} {'10-Year':>15s}")
print("-" * 60)
print(f"{'Total rows':<30s} {df_3year.shape[0]:>15,} {df_10year.shape[0]:>15,}")
print(f"{'Total columns':<30s} {df_3year.shape[1]:>15,} {df_10year.shape[1]:>15,}")
print(f"{'Class 0 count':<30s} {target_dist_3year.get(0, 0):>15,} {target_dist_10year.get(0.0, 0):>15,}")
print(f"{'Class 1 count':<30s} {target_dist_3year.get(1, 0):>15,} {target_dist_10year.get(1.0, 0):>15,}")
print(f"{'Class 1 percentage':<30s} {target_dist_3year.get(1, 0)/len(df_3year)*100:>14.2f}% {target_dist_10year.get(1.0, 0)/len(df_10year)*100:>14.2f}%")
print(f"{'Memory usage (MB)':<30s} {val_3year['memory_usage_mb']:>14.2f} {val_10year['memory_usage_mb']:>14.2f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {output_dir.absolute()}")
