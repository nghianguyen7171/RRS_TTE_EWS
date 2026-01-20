#!/usr/bin/env python3
"""
Problem Definition Analysis
Determines the appropriate problem formulation based on EDA findings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_3year_dataset, load_10year_dataset, get_feature_groups

print("=" * 80)
print("PROBLEM DEFINITION ANALYSIS")
print("=" * 80)

# Load datasets
df_3year = load_3year_dataset("../data/CNUH_3Y.csv")
df_10year = load_10year_dataset("../data/10yrs_proc.csv")

# ============================================================================
# 1. PROBLEM TYPE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("1. PROBLEM TYPE DETERMINATION")
print("=" * 80)

# Check target variable
print("\nTarget Variable Analysis:")
print(f"3-year dataset: {df_3year['target'].nunique()} unique values")
print(f"10-year dataset: {df_10year['target'].nunique()} unique values")

target_3year = df_3year['target'].value_counts().sort_index()
target_10year = df_10year['target'].value_counts().sort_index()

print("\n3-year target distribution:")
for val, count in target_3year.items():
    print(f"  Class {val}: {count:,} ({count/len(df_3year)*100:.2f}%)")

print("\n10-year target distribution:")
for val, count in target_10year.items():
    print(f"  Class {val}: {count:,} ({count/len(df_10year)*100:.2f}%)")

# Determine problem type
print("\n" + "-" * 80)
print("PROBLEM TYPE: BINARY CLASSIFICATION")
print("-" * 80)
print("✓ Target is binary (0 = normal, 1 = deterioration)")
print("✓ Highly imbalanced dataset (class imbalance issue)")
print("✓ Classification problem: predict if patient will deteriorate")

# ============================================================================
# 2. CLASS IMBALANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. CLASS IMBALANCE ANALYSIS")
print("=" * 80)

imbalance_ratio_3year = target_3year[0] / target_3year[1] if 1 in target_3year else float('inf')
imbalance_ratio_10year = target_10year[0.0] / target_10year[1.0] if 1.0 in target_10year else float('inf')

print(f"\n3-year dataset imbalance ratio: {imbalance_ratio_3year:.1f}:1")
print(f"10-year dataset imbalance ratio: {imbalance_ratio_10year:.1f}:1")

print("\nImbalance Severity:")
if imbalance_ratio_3year > 100:
    print("  3-year: EXTREMELY IMBALANCED (>100:1)")
elif imbalance_ratio_3year > 10:
    print("  3-year: HIGHLY IMBALANCED (10-100:1)")
else:
    print("  3-year: MODERATELY IMBALANCED (<10:1)")

if imbalance_ratio_10year > 100:
    print("  10-year: EXTREMELY IMBALANCED (>100:1)")
elif imbalance_ratio_10year > 10:
    print("  10-year: HIGHLY IMBALANCED (10-100:1)")
else:
    print("  10-year: MODERATELY IMBALANCED (<10:1)")

print("\nRecommendations:")
print("  - Use appropriate metrics: Precision, Recall, F1-score, AUC-ROC")
print("  - Consider class weighting or resampling techniques")
print("  - Use stratified cross-validation")

# ============================================================================
# 3. TEMPORAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. TEMPORAL PATTERN ANALYSIS")
print("=" * 80)

# Check temporal features
print("\nTemporal features available:")
print("  3-year dataset:")
if 'measurement_time' in df_3year.columns:
    print(f"    - measurement_time: {df_3year['measurement_time'].notna().sum():,} values")
if 'event_time' in df_3year.columns:
    print(f"    - event_time: {df_3year['event_time'].notna().sum():,} values")
if 'detection_time' in df_3year.columns:
    print(f"    - detection_time: {df_3year['detection_time'].notna().sum():,} values")

print("\n  10-year dataset:")
if 'measurement_time' in df_10year.columns:
    print(f"    - measurement_time: {df_10year['measurement_time'].notna().sum():,} values")
if 'adjusted_time' in df_10year.columns:
    print(f"    - adjusted_time: {df_10year['adjusted_time'].notna().sum():,} values")
if 'event_time' in df_10year.columns:
    print(f"    - event_time: {df_10year['event_time'].notna().sum():,} values")
if 'detection_time' in df_10year.columns:
    print(f"    - detection_time: {df_10year['detection_time'].notna().sum():,} values")

print("\nTemporal Problem Formulation:")
print("  - Current: Static classification (single time point prediction)")
print("  - Potential: Time-series prediction (predict deterioration X hours before event)")
print("  - Recommendation: Start with static classification, explore time-series later")

# ============================================================================
# 4. FEATURE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. FEATURE ANALYSIS")
print("=" * 80)

features_3year = get_feature_groups(df_3year)
features_10year = get_feature_groups(df_10year)

print("\nFeature Categories:")
print(f"  3-year dataset:")
print(f"    - Vital signs: {len(features_3year['vital_signs'])} features")
print(f"    - Lab values: {len(features_3year['lab_values'])} features")
print(f"    - Demographics: {len(features_3year['demographics'])} features")

print(f"\n  10-year dataset:")
print(f"    - Vital signs: {len(features_10year['vital_signs'])} features")
print(f"    - Lab values: {len(features_10year['lab_values'])} features")
print(f"    - Demographics: {len(features_10year['demographics'])} features")

print("\nFeature Engineering Recommendations:")
print("  - Handle missing values (imputation or indicator variables)")
print("  - Normalize/standardize numeric features")
print("  - Create time-based features (time since admission, time to event)")
print("  - Consider feature interactions (vital signs × lab values)")

# ============================================================================
# 5. DATA SPLIT STRATEGY
# ============================================================================
print("\n" + "=" * 80)
print("5. DATA SPLIT STRATEGY")
print("=" * 80)

print("\nRecommended Split Strategy:")
print("  - Use stratified train/validation/test split")
print("  - Maintain class distribution in each split")
print("  - Consider patient-level split (avoid data leakage)")
print("  - Suggested ratios: 70% train, 15% validation, 15% test")

# Check for patient-level data
if 'Patient' in df_3year.columns:
    unique_patients_3year = df_3year['Patient'].nunique()
    print(f"\n3-year dataset: {unique_patients_3year:,} unique patients")
    print(f"  Average records per patient: {len(df_3year)/unique_patients_3year:.1f}")

if 'Patient' in df_10year.columns:
    unique_patients_10year = df_10year['Patient'].nunique()
    print(f"\n10-year dataset: {unique_patients_10year:,} unique patients")
    print(f"  Average records per patient: {len(df_10year)/unique_patients_10year:.1f}")

print("\n⚠️  IMPORTANT: Use patient-level splitting to avoid data leakage")
print("   (Same patient should not appear in both train and test sets)")

# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================
print("\n" + "=" * 80)
print("6. EVALUATION METRICS")
print("=" * 80)

print("\nRecommended Metrics for Imbalanced Classification:")
print("  Primary Metrics:")
print("    - AUC-ROC: Overall model performance")
print("    - Precision: Minimize false positives (important for clinical decisions)")
print("    - Recall (Sensitivity): Minimize false negatives (catch all deteriorations)")
print("    - F1-Score: Balance between precision and recall")
print("    - Specificity: True negative rate")

print("\n  Secondary Metrics:")
print("    - Confusion Matrix: Detailed performance breakdown")
print("    - Precision-Recall Curve: Better for imbalanced data")
print("    - ROC Curve: Visualize threshold selection")

print("\n  Clinical Relevance:")
print("    - High Recall is critical (don't miss deteriorations)")
print("    - Precision should be reasonable (avoid unnecessary alerts)")
print("    - Consider cost-sensitive learning")

# ============================================================================
# 7. BASELINE MODEL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("7. BASELINE MODEL RECOMMENDATIONS")
print("=" * 80)

print("\nSuggested Model Progression:")
print("  1. Baseline Models:")
print("     - Random Forest (handles missing values, feature importance)")
print("     - XGBoost (handles imbalance, good performance)")
print("     - Logistic Regression (interpretable, baseline)")

print("\n  2. Advanced Models:")
print("     - LightGBM (fast, handles large datasets)")
print("     - Neural Networks (if sufficient data)")
print("     - Ensemble methods (combine multiple models)")

print("\n  3. Handling Imbalance:")
print("     - Class weights in model")
print("     - SMOTE or ADASYN oversampling")
print("     - Undersampling majority class")
print("     - Cost-sensitive learning")

# ============================================================================
# 8. FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("8. FINAL PROBLEM DEFINITION")
print("=" * 80)

print("\n" + "=" * 80)
print("PROBLEM TYPE: BINARY CLASSIFICATION")
print("=" * 80)
print("\nObjective:")
print("  Predict clinical deterioration (cardiac arrest/endotracheal intubation)")
print("  using vital signs and laboratory test results")

print("\nKey Characteristics:")
print("  ✓ Binary target: 0 (normal) vs 1 (deterioration)")
print("  ✓ Highly imbalanced: ~0.3% (3-year) to ~4.7% (10-year) positive class")
print("  ✓ Static prediction: Single time point classification")
print("  ✓ Feature-rich: Vital signs + Lab values + Demographics")
print("  ✓ Temporal data available: Can explore time-series prediction later")

print("\nRecommended Approach:")
print("  1. Start with static binary classification")
print("  2. Use stratified patient-level train/validation/test split")
print("  3. Handle class imbalance with appropriate techniques")
print("  4. Focus on Recall (sensitivity) for clinical safety")
print("  5. Use tree-based models (Random Forest, XGBoost) as baseline")
print("  6. Evaluate with AUC-ROC, Precision, Recall, F1-score")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
