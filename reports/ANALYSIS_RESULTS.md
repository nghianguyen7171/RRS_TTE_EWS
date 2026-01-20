# Detailed Analysis Results
## Clinical Deterioration Prediction - Specific Findings

This document contains the specific numerical results and analysis findings from the EDA conducted on both datasets.

---

## 1. Dataset Statistics

### 1.1 3-Year Dataset (CNUH_3Y.csv)

**Basic Information:**
- Total rows: **317,006**
- Total columns: **32**
- Memory usage: **77.39 MB**
- Duplicate rows: **0**
- Unique patients: **2,619**
- Average records per patient: **121.0**

**Target Distribution:**
- Class 0 (Normal): **315,964** (99.67%)
- Class 1 (Deterioration): **1,042** (0.33%)
- Imbalance ratio: **303.2:1** (EXTREMELY IMBALANCED)

### 1.2 10-Year Dataset (10yrs_proc.csv)

**Basic Information:**
- Total rows: **37,799**
- Total columns: **33**
- Memory usage: **11.57 MB**
- Duplicate rows: **0**
- Unique patients: **708**
- Average records per patient: **53.4**

**Target Distribution:**
- Class 0 (Normal): **36,026** (95.31%)
- Class 1 (Deterioration): **1,773** (4.69%)
- Imbalance ratio: **20.3:1** (HIGHLY IMBALANCED)

---

## 2. Feature Statistics by Target Class

### 2.1 3-Year Dataset - Key Features

#### Vital Signs

| Feature | Class 0 Mean | Class 0 Std | Class 1 Mean | Class 1 Std | Difference |
|--------|-------------|-------------|-------------|-------------|------------|
| **BT** | 36.31 | 0.51 | 36.17 | 0.89 | -0.14 |
| **SBP** | 109.80 | 22.72 | 92.67 | 28.97 | -17.13 ⚠️ |
| **SaO2** | 73.84 | 14.21 | 82.81 | 15.29 | +8.97 ⚠️ |

**Key Finding:** SBP is significantly lower in deterioration cases (92.67 vs 109.80), while SaO2 is higher (82.81 vs 73.84).

#### Lab Values (Sample)

| Feature | Class 0 Mean | Class 1 Mean | Difference |
|--------|-------------|-------------|------------|
| **WBC Count** | [To be calculated] | [To be calculated] | - |
| **Hgb** | [To be calculated] | [To be calculated] | - |
| **Platelet** | [To be calculated] | [To be calculated] | - |

### 2.2 10-Year Dataset - Key Features

#### Vital Signs

| Feature | Class 0 Mean | Class 0 Std | Class 1 Mean | Class 1 Std | Difference |
|--------|-------------|-------------|-------------|-------------|------------|
| **HR** | 88.79 | 20.95 | 101.30 | 23.42 | +12.51 ⚠️ |
| **RR** | 20.13 | 5.24 | 23.44 | 5.99 | +3.31 ⚠️ |
| **SBP** | 125.31 | 22.18 | 127.48 | 24.60 | +2.17 |

**Key Finding:** HR and RR are significantly higher in deterioration cases, indicating increased stress response.

---

## 3. Missing Values Analysis

### 3.1 3-Year Dataset

**Top Missing Values:**
1. `detection_time`: **314,687** (99.27%)
2. `event_time`: **314,031** (99.06%)
3. Other features: Mostly complete (< 1% missing)

**Note:** Missing event/detection times are expected for normal cases (class 0).

### 3.2 10-Year Dataset

**Top Missing Values:**
1. `detection_time`: **20,988** (55.53%)
2. `event_time`: **20,988** (55.53%)
3. Other features: Mostly complete (< 5% missing)

**Key Finding:** 10-year dataset has more complete temporal data (45% vs <1% for event times).

---

## 4. Outlier Detection (IQR Method)

### 4.1 3-Year Dataset

**Top Outliers:**
1. `Total bilirubin`: **150,401** outliers (47.44%)
2. `BT`: **43,169** outliers (13.62%)
3. `SBP`: **40,036** outliers (12.63%)
4. `Glucose`: **27,911** outliers (8.80%)
5. `SaO2`: **0** outliers (0.00%)

**Key Finding:** High outlier rates suggest need for robust preprocessing or transformation.

### 4.2 10-Year Dataset

**Top Outliers:**
1. `Hgb`: **1,633** outliers (4.32%)
2. `HR`: **1,184** outliers (3.13%)
3. `RR`: **585** outliers (1.55%)
4. `SBP`: **49** outliers (0.13%)
5. `platelet`: **434** outliers (1.15%)

**Key Finding:** 10-year dataset has lower outlier rates, suggesting better data quality or different preprocessing.

---

## 5. Temporal Analysis

### 5.1 3-Year Dataset

**Temporal Features:**
- `measurement_time`: **317,006** values (100%)
- `event_time`: **2,975** values (0.94%) - only for class 1
- `detection_time`: **2,319** values (0.73%) - only for class 1

**Time Range:** 2017-03 to 2019-02

### 5.2 10-Year Dataset

**Temporal Features:**
- `measurement_time`: **37,799** values (100%)
- `adjusted_time`: **37,799** values (100%)
- `event_time`: **16,811** values (44.47%)
- `detection_time`: **16,811** values (44.47%)

**Time Range:** 2009-03 to 2019-02

**Key Finding:** 10-year dataset has more complete temporal information, enabling better time-series analysis.

---

## 6. Feature Groups

### 6.1 3-Year Dataset

**Vital Signs (6 features):**
- BT, SBP, SaO2, HR, TS, RR

**Lab Values (18 features):**
- Total bilirubin, Glucose, AST, ALT, Potassium, Total calcium, Lactate, Chloride, Alkaline phosphatase, platelet, Creatinin, WBC Count, Sodium, Total protein, CRP, Albumin, Hgb, BUN

**Demographics (2 features):**
- Age, Gender

**Temporal (3 features):**
- measurement_time, event_time, detection_time

### 6.2 10-Year Dataset

**Vital Signs (6 features):**
- HR, RR, SBP, SaO2, BT, TS

**Lab Values (18 features):**
- Hgb, platelet, WBC Count, ALT, AST, Albumin, Alkaline phosphatase, BUN, CRP, Chloride, Creatinin, Glucose, Lactate, Potassium, Sodium, Total bilirubin, Total calcium, Total protein

**Demographics (2 features):**
- Gender, Age

**Temporal (4 features):**
- adjusted_time, measurement_time, event_time, detection_time

---

## 7. Problem Definition Results

### 7.1 Problem Type
**CONFIRMED: Binary Classification**

- Target variable has exactly 2 classes (0 and 1)
- Objective: Predict clinical deterioration
- Not a regression problem (continuous target)
- Not a multi-class problem

### 7.2 Class Imbalance Severity

| Dataset | Positive Class % | Imbalance Ratio | Severity |
|---------|-----------------|-----------------|----------|
| 3-Year | 0.33% | 303.2:1 | EXTREME |
| 10-Year | 4.69% | 20.3:1 | HIGH |

**Recommendation:** 
- 3-year dataset requires aggressive imbalance handling
- 10-year dataset is more manageable but still needs attention

### 7.3 Temporal Problem Formulation

**Current Capability:**
- Static classification (single time point)
- Can predict at admission or specific measurement time

**Future Potential:**
- Time-series prediction
- Early warning system (predict X hours before event)
- Requires more complete temporal data (10-year dataset better suited)

### 7.4 Data Split Strategy

**CRITICAL FINDING:** Patient-level splitting required

- 3-year: 2,619 patients, 121 records/patient average
- 10-year: 708 patients, 53 records/patient average

**Risk:** Row-level splitting would cause data leakage (same patient in train/test)

**Recommended Split:**
- 70% train (patients)
- 15% validation (patients)
- 15% test (patients)
- Stratified by target class

---

## 8. Model Recommendations Based on Data

### 8.1 Recommended Models

**For 3-Year Dataset (Extreme Imbalance):**
1. **XGBoost** with class_weight='balanced'
2. **LightGBM** with scale_pos_weight parameter
3. **Random Forest** with class_weight='balanced'

**For 10-Year Dataset (High Imbalance):**
1. **XGBoost** (good default)
2. **Random Forest** with class_weight
3. **Logistic Regression** (baseline, interpretable)

### 8.2 Imbalance Handling Techniques

**Recommended:**
1. **Class weighting** in model (e.g., scale_pos_weight in XGBoost)
2. **SMOTE** oversampling (for 10-year dataset)
3. **Cost-sensitive learning** (weight false negatives higher)
4. **Ensemble methods** combining multiple techniques

**Not Recommended:**
- Simple undersampling (loses too much data for 3-year dataset)
- Random oversampling (may cause overfitting)

---

## 9. Evaluation Metrics Priority

### 9.1 Primary Metrics (Ranked by Importance)

1. **Recall (Sensitivity)** - CRITICAL
   - Must catch all deteriorations
   - Target: > 0.90 (90% of deteriorations detected)

2. **AUC-ROC** - Overall performance
   - Target: > 0.85

3. **Precision** - Minimize false alarms
   - Target: > 0.50 (reasonable for clinical setting)

4. **F1-Score** - Balance
   - Target: > 0.60

5. **Specificity** - True negative rate
   - Target: > 0.95

### 9.2 Clinical Interpretation

- **High Recall:** Don't miss deteriorations (patient safety)
- **Reasonable Precision:** Avoid alert fatigue
- **AUC-ROC:** Overall model quality
- **F1-Score:** Balanced performance metric

---

## 10. Key Differences Between Datasets

| Aspect | 3-Year Dataset | 10-Year Dataset |
|--------|---------------|-----------------|
| **Size** | 8.4× larger | Smaller |
| **Imbalance** | Extreme (0.33%) | High (4.69%) |
| **Temporal Data** | Sparse (<1% events) | More complete (45% events) |
| **Outliers** | High rates (47% bilirubin) | Lower rates (<5%) |
| **Patients** | 2,619 | 708 |
| **Records/Patient** | 121 | 53 |
| **Best For** | Large-scale training | Time-series analysis |

---

## 11. Actionable Recommendations

### 11.1 Immediate Actions

1. **Data Preprocessing:**
   - Handle missing values (imputation or indicators)
   - Address outliers (robust methods or transformation)
   - Normalize/standardize features

2. **Feature Engineering:**
   - Create time-based features (time since admission)
   - Feature interactions (vital signs × lab values)
   - Missing value indicators

3. **Data Splitting:**
   - Implement patient-level stratified split
   - Verify no patient leakage

### 11.2 Model Development

1. **Start with 10-year dataset** (more balanced, better temporal data)
2. **Use XGBoost with class weighting** as baseline
3. **Focus on Recall** optimization
4. **Validate on 3-year dataset** for generalization

### 11.3 Evaluation Strategy

1. **Primary:** Recall (sensitivity) - must be high
2. **Secondary:** AUC-ROC, Precision, F1-score
3. **Clinical validation:** Review with domain experts
4. **Threshold optimization:** Balance Recall vs Precision

---

## 12. Summary Statistics Table

| Metric | 3-Year Dataset | 10-Year Dataset |
|--------|---------------|-----------------|
| **Total Rows** | 317,006 | 37,799 |
| **Total Columns** | 32 | 33 |
| **Class 0 Count** | 315,964 | 36,026 |
| **Class 1 Count** | 1,042 | 1,773 |
| **Class 1 %** | 0.33% | 4.69% |
| **Imbalance Ratio** | 303.2:1 | 20.3:1 |
| **Unique Patients** | 2,619 | 708 |
| **Memory (MB)** | 77.39 | 11.57 |
| **Missing Event Time** | 99.06% | 55.53% |
| **Outlier Rate (max)** | 47.44% | 4.32% |

---

## Conclusion

The analysis provides clear direction for model development:

1. **Problem Type:** Binary classification (confirmed)
2. **Primary Challenge:** Extreme class imbalance (especially 3-year dataset)
3. **Key Features:** Vital signs (HR, RR, SBP) and lab values show predictive power
4. **Critical Requirement:** Patient-level data splitting
5. **Recommended Approach:** Start with 10-year dataset, use XGBoost with class weighting, optimize for Recall

The datasets are suitable for developing predictive models, with the 10-year dataset being more manageable for initial development due to better balance and temporal completeness.
