# Exploratory Data Analysis Summary Report
## Clinical Deterioration Prediction

**Date:** Generated from analysis  
**Datasets:** CNUH_3Y.csv (3-year) and 10yrs_proc.csv (10-year)

---

## Executive Summary

This report summarizes the exploratory data analysis (EDA) conducted on two clinical datasets for predicting patient deterioration. The analysis reveals highly imbalanced binary classification problems with distinct characteristics between the two datasets.

---

## 1. Dataset Overview

### 1.1 Dataset Sizes

| Metric | 3-Year Dataset | 10-Year Dataset |
|--------|---------------|-----------------|
| **Total Rows** | 317,006 | 37,799 |
| **Total Columns** | 32 | 33 |
| **Memory Usage** | 77.39 MB | 11.57 MB |
| **Unique Patients** | 2,619 | 708 |
| **Avg Records/Patient** | 121.0 | 53.4 |

### 1.2 Target Distribution

#### 3-Year Dataset
- **Class 0 (Normal):** 315,964 (99.67%)
- **Class 1 (Deterioration):** 1,042 (0.33%)
- **Imbalance Ratio:** 303.2:1 (EXTREMELY IMBALANCED)

#### 10-Year Dataset
- **Class 0 (Normal):** 36,026 (95.31%)
- **Class 1 (Deterioration):** 1,773 (4.69%)
- **Imbalance Ratio:** 20.3:1 (HIGHLY IMBALANCED)

**Key Finding:** Both datasets are highly imbalanced, with the 3-year dataset being extremely imbalanced. The 10-year dataset has a more balanced distribution (4.69% vs 0.33% positive class).

---

## 2. Feature Analysis

### 2.1 Feature Categories

Both datasets contain similar feature structures:

| Category | Count | Examples |
|----------|-------|----------|
| **Vital Signs** | 6 | HR, RR, SBP, SaO2, BT, TS |
| **Lab Values** | 18 | WBC, Hgb, platelet, ALT, AST, Albumin, BUN, CRP, etc. |
| **Demographics** | 2 | Age, Gender |
| **Temporal** | 3-4 | measurement_time, event_time, detection_time |

### 2.2 Missing Values

#### 3-Year Dataset (Top Missing)
- `detection_time`: 314,687 (99.27%)
- `event_time`: 314,031 (99.06%)

#### 10-Year Dataset (Top Missing)
- `detection_time`: 20,988 (55.53%)
- `event_time`: 20,988 (55.53%)

**Note:** Missing event/detection times are expected for normal cases (class 0).

### 2.3 Outlier Detection (IQR Method)

#### 3-Year Dataset
- `Total bilirubin`: 150,401 outliers (47.44%)
- `BT`: 43,169 outliers (13.62%)
- `SBP`: 40,036 outliers (12.63%)
- `Glucose`: 27,911 outliers (8.80%)

#### 10-Year Dataset
- `Hgb`: 1,633 outliers (4.32%)
- `HR`: 1,184 outliers (3.13%)
- `RR`: 585 outliers (1.55%)
- `platelet`: 434 outliers (1.15%)

---

## 3. Statistical Analysis by Target Class

### 3.1 Key Differences (3-Year Dataset)

| Feature | Class 0 Mean | Class 1 Mean | Difference |
|---------|-------------|--------------|------------|
| **SBP** | 109.8 | 92.7 | Lower in deterioration |
| **SaO2** | 73.8 | 82.8 | Higher in deterioration |
| **BT** | 36.3 | 36.2 | Similar |

### 3.2 Key Differences (10-Year Dataset)

| Feature | Class 0 Mean | Class 1 Mean | Difference |
|---------|-------------|--------------|------------|
| **HR** | 88.8 | 101.3 | Higher in deterioration |
| **RR** | 20.1 | 23.4 | Higher in deterioration |
| **SBP** | 125.3 | 127.5 | Similar |

**Key Finding:** Vital signs show meaningful differences between classes, suggesting they are predictive features.

---

## 4. Problem Definition

### 4.1 Problem Type
**BINARY CLASSIFICATION**
- Predict clinical deterioration (cardiac arrest/endotracheal intubation)
- Target: 0 (normal) vs 1 (deterioration)

### 4.2 Key Characteristics
- ✓ Binary target variable
- ✓ Highly imbalanced datasets
- ✓ Static prediction (single time point)
- ✓ Feature-rich (vital signs + lab values + demographics)
- ✓ Temporal data available for future time-series exploration

### 4.3 Recommended Approach

1. **Start with static binary classification**
2. **Use stratified patient-level train/validation/test split**
   - 70% train, 15% validation, 15% test
   - **CRITICAL:** Patient-level split to avoid data leakage
3. **Handle class imbalance:**
   - Class weighting in models
   - SMOTE/ADASYN oversampling
   - Cost-sensitive learning
4. **Focus on Recall (sensitivity)** for clinical safety
5. **Use tree-based models** (Random Forest, XGBoost) as baseline
6. **Evaluate with multiple metrics:**
   - AUC-ROC (primary)
   - Precision, Recall, F1-score
   - Precision-Recall Curve

---

## 5. Data Split Strategy

### 5.1 Patient-Level Splitting

**IMPORTANT:** Use patient-level splitting to avoid data leakage.

- **3-year dataset:** 2,619 unique patients
- **10-year dataset:** 708 unique patients

**Recommendation:**
- Split by patient ID, not by rows
- Ensure same patient doesn't appear in both train and test sets
- Maintain class distribution in each split (stratified)

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics
- **AUC-ROC:** Overall model performance
- **Precision:** Minimize false positives
- **Recall (Sensitivity):** Minimize false negatives (CRITICAL)
- **F1-Score:** Balance between precision and recall
- **Specificity:** True negative rate

### 6.2 Clinical Relevance
- **High Recall is critical:** Don't miss deteriorations
- **Precision should be reasonable:** Avoid unnecessary alerts
- **Consider cost-sensitive learning:** Weight false negatives higher

---

## 7. Model Recommendations

### 7.1 Baseline Models
1. **Random Forest**
   - Handles missing values
   - Provides feature importance
   - Good for imbalanced data

2. **XGBoost**
   - Excellent for imbalanced data
   - Built-in class weighting
   - High performance

3. **Logistic Regression**
   - Interpretable
   - Good baseline
   - Fast training

### 7.2 Advanced Models
- LightGBM (fast, handles large datasets)
- Neural Networks (if sufficient data)
- Ensemble methods

### 7.3 Handling Imbalance
- Class weights in model
- SMOTE or ADASYN oversampling
- Undersampling majority class
- Cost-sensitive learning

---

## 8. Feature Engineering Recommendations

1. **Handle missing values:**
   - Imputation (median, mean, mode)
   - Indicator variables for missingness

2. **Normalize/standardize:**
   - Numeric features (vital signs, lab values)
   - Use StandardScaler or MinMaxScaler

3. **Create time-based features:**
   - Time since admission
   - Time to event (for positive cases)
   - Time to detection

4. **Consider feature interactions:**
   - Vital signs × lab values
   - Age × lab values
   - Gender-specific patterns

---

## 9. Key Findings Summary

### 9.1 Dataset Comparison

| Aspect | 3-Year Dataset | 10-Year Dataset |
|--------|---------------|-----------------|
| **Size** | 8.4× larger | Smaller but more balanced |
| **Imbalance** | Extreme (0.33%) | High (4.69%) |
| **Temporal Coverage** | 2017-2019 | 2009-2019 |
| **Patients** | 2,619 | 708 |

### 9.2 Critical Issues
1. **Extreme class imbalance** in 3-year dataset
2. **Missing values** in temporal features (expected for normal cases)
3. **Outliers** in several features (especially 3-year dataset)
4. **Patient-level data leakage risk** (must use patient-level splitting)

### 9.3 Opportunities
1. **Rich feature set** (vital signs + lab values)
2. **Temporal data** available for time-series exploration
3. **Clear class separation** in some features
4. **Multiple datasets** for validation

---

## 10. Next Steps

1. **Data Preprocessing:**
   - Handle missing values
   - Remove/transform outliers
   - Normalize features

2. **Feature Engineering:**
   - Create time-based features
   - Feature interactions
   - Dimensionality reduction (if needed)

3. **Model Development:**
   - Implement patient-level splitting
   - Train baseline models
   - Handle class imbalance
   - Hyperparameter tuning

4. **Evaluation:**
   - Comprehensive metric evaluation
   - Feature importance analysis
   - Model interpretation
   - Clinical validation

5. **Future Exploration:**
   - Time-series prediction
   - Early warning system (predict X hours before event)
   - Multi-task learning

---

## Conclusion

The analysis reveals a challenging but feasible binary classification problem. The extreme class imbalance requires careful handling, but the rich feature set and clear differences between classes suggest that predictive models can be developed. The recommended approach is to start with static binary classification using tree-based models with appropriate imbalance handling techniques.

**Problem Type:** Binary Classification  
**Primary Challenge:** Extreme class imbalance  
**Recommended Approach:** Static classification with patient-level splitting and imbalance handling  
**Key Metrics:** AUC-ROC, Recall (critical), Precision, F1-score
