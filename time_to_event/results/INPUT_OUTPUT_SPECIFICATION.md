# Time-to-Event Survival Prediction: Input/Output Specification

**Problem Type:** Survival Analysis / Time-to-Event Prediction  
**Task:** Predict time until clinical deterioration (cardiac arrest/endotracheal intubation)  
**Analysis Date:** January 2025

---

## Problem Overview

This is a **survival analysis** problem where we predict the time until a clinical deterioration event occurs, given patient features at a specific time point during hospitalization. The problem involves **right-censored data**, meaning many patients do not experience the event during the observation period.

---

## INPUT (X - Features)

### Feature Dimensions
- **Total Features:** 57 features per observation
- **Data Format:** Normalized numerical features (standardized)
- **Shape:** `(n_observations, 57)`
  - 3-year dataset: 317,006 observations
  - 10-year dataset: 37,799 observations

### Feature Categories

#### 1. **Vital Signs** (Current Measurements)
- Heart Rate (HR)
- Systolic Blood Pressure (SBP)
- Oxygen Saturation (SaO2)
- Respiration Rate (RR)
- Body Temperature (BT/TS)

#### 2. **Vital Sign Historical Statistics**
- Mean of previous measurements: `SBP_mean_prev`, `HR_mean_prev`, `SaO2_mean_prev`, `RR_mean_prev`, `BT_mean_prev`
- Standard deviation of previous measurements: `SBP_std_prev`, `HR_std_prev`, `SaO2_std_prev`, `RR_std_prev`, `BT_std_prev`
- Trends: `HR_trend`, `SBP_trend`, etc.

#### 3. **Laboratory Values**

**Complete Blood Count (CBC):**
- White Blood Cell Count (WBC Count)
- Hemoglobin (Hgb)
- Platelet count

**Chemistry Panel:**
- Glucose
- Sodium
- Potassium
- Chloride
- Creatinine
- Blood Urea Nitrogen (BUN)

**Liver Function Tests:**
- Aspartate Aminotransferase (AST)
- Alanine Aminotransferase (ALT)
- Total Bilirubin
- Alkaline Phosphatase

**Other Laboratory Values:**
- C-Reactive Protein (CRP) - inflammatory marker
- Albumin - nutritional status
- Total Protein
- Total Calcium
- Lactate

#### 4. **Demographics**
- Age
- Gender

#### 5. **Temporal Features**
- `hour_of_day` - Hour of day (0-23)
- `day_of_week` - Day of week (0-6)
- `is_weekend` - Binary indicator (0/1)
- `hour_sin`, `hour_cos` - Circular encoding of hour (captures circadian rhythms)
- `day_sin` - Circular encoding of day of week
- `time_since_admission_hours` - Time elapsed since hospital admission

#### 6. **Interaction Features**
- Vital Sign × Laboratory interactions:
  - `HR_x_Hgb`, `HR_x_platelet`, `HR_x_WBC Count`
  - `RR_x_Hgb`, `RR_x_platelet`, `RR_x_WBC Count`
  - `SBP_x_Hgb`, `SBP_x_Glucose`, `SBP_x_platelet`
  - `SaO2_x_AST`, `SaO2_x_Glucose`
  - `BT_x_AST`, `BT_x_Total bilirubin`
- Age × Laboratory interactions:
  - `Age_x_AST`, `Age_x_ALT`, `Age_x_Total bilirubin`

### Data Preprocessing
- **Normalization:** All features are standardized (mean=0, std=1)
- **Missing Values:** Handled via imputation or exclusion
- **Outliers:** Detected and handled using IQR method
- **Temporal Alignment:** Features aligned to measurement time points

### Data Location
- **3-year dataset:** `time_to_event/data/processed_3year/X_features.csv`
- **10-year dataset:** `time_to_event/data/processed_10year/X_features.csv`

---

## OUTPUT (Y - Survival Labels)

### Two-Component Output

For each observation, the model requires **two outputs**:

#### 1. **`y_time` (Time-to-Event)**
- **Definition:** Time (in hours) from the current measurement to either:
  - The event (if `y_event = 1`), OR
  - Censoring time (if `y_event = 0`)
- **Type:** Continuous, non-negative float
- **Range:**
  - 3-year dataset: 0.0 to 18,795 hours
  - 10-year dataset: 0.0 to 7,584 hours
- **Units:** Hours
- **Location:** `time_to_event/data/processed_*/y_time.npy`

**Examples:**
- `y_time = 48.5` means the event occurred (or observation ended) 48.5 hours after this measurement
- `y_time = 720.0` means the observation ended 720 hours (30 days) after this measurement

#### 2. **`y_event` (Event Indicator)**
- **Definition:** Binary indicator of whether the event occurred
- **Type:** Binary integer (0 or 1)
- **Values:**
  - `0` = **Censored** (patient did not experience event during observation period)
    - Patient discharged without event
    - End of observation period reached
    - Lost to follow-up
  - `1` = **Event Occurred** (clinical deterioration happened)
    - Cardiac arrest occurred
    - Endotracheal intubation performed
- **Location:** `time_to_event/data/processed_*/y_event.npy`

### Event Definition
**Event:** Cardiac arrest or endotracheal intubation during hospitalization

**Censoring:** Patient did not experience the event during the observation period (discharged, end of study, or lost to follow-up)

### Data Characteristics

| Dataset | Observations | Events | Censored | Event Rate | Median Time (events) | Median Time (censored) |
|---------|-------------|--------|----------|------------|---------------------|----------------------|
| 3-year  | 317,006     | 1,042  | 315,964  | 0.33%      | 24 hours            | 103 hours            |
| 10-year | 37,799      | 1,773  | 36,026   | 4.69%      | 19 hours            | 35 hours             |

**Key Observations:**
- **Severe class imbalance:** Event rates are very low (0.33-4.69%)
- **Right-censoring:** Most patients are censored (>95%)
- **Temporal structure:** Each observation is a time point in a patient's hospitalization
- **Longitudinal data:** Multiple observations per patient (mean: 121 obs/patient in 3-year, 53 in 10-year)

---

## Data Structure

### Format
```
For each observation i:
  Input:  X[i] = [feature_1, feature_2, ..., feature_57]  (57 normalized features)
  Output: Y[i] = (y_time[i], y_event[i])
           where:
           - y_time[i]  = time until event/censoring (hours, continuous)
           - y_event[i] = 1 if event occurred, 0 if censored (binary)
```

### Example
**Input (X):**
- Patient's current vital signs: HR=85, SBP=120, SaO2=98%, RR=18, BT=36.5°C
- Laboratory values: CRP=5.2 mg/L, Albumin=3.5 g/dL, AST=45 U/L
- Temporal features: hour_of_day=14, day_of_week=3, time_since_admission=48 hours
- Historical statistics: SBP_mean_prev=125, SBP_std_prev=8.5
- Interaction features: HR_x_Hgb=12.5, SBP_x_Glucose=600
- Demographics: Age=65, Gender=1
- ... (57 total features, all normalized)

**Output (Y):**
- `y_time = 24.5` hours
- `y_event = 1` (event occurred 24.5 hours after this measurement)

---

## Model Outputs (Predictions)

### Cox Proportional Hazards Models Produce:

#### 1. **Hazard Ratio (HR)** for Each Feature
- **Interpretation:**
  - HR > 1: Feature increases risk of event
  - HR < 1: Feature decreases risk (protective factor)
  - HR = 1: No effect
- **Example:** HR(CRP) = 3.31 means each unit increase in CRP triples the hazard

#### 2. **Risk Score** (Linear Combination)
```
risk_score = β₁×feature₁ + β₂×feature₂ + ... + βₙ×featureₙ
```
- Higher risk score = higher risk of event
- Used for risk stratification (Low, Medium-Low, Medium-High, High)

#### 3. **Survival Probability** Over Time
```
S(t|X) = S₀(t)^exp(risk_score)
```
- Probability of surviving beyond time `t` given features `X`
- `S₀(t)` = baseline survival function
- Produces survival curves showing probability over time

#### 4. **Predicted Time-to-Event** (Optional)
- Expected time until event based on risk score
- Can be used for early warning systems

### Model Performance Metrics

- **Concordance (C-index):** Measures discrimination ability (0.5 = random, 1.0 = perfect)
  - Current best: 0.885 (3-year), 0.778 (10-year)
- **AIC/BIC:** Measures model fit (lower is better)
  - Current best: AIC = 22,074 (3-year), 34,044 (10-year) for Stepwise model
- **Hazard Ratios:** Effect sizes for each feature
- **P-values:** Statistical significance of features

---

## Data Flow

### Training Phase
```
Raw Data → Preprocessing → Feature Engineering → Normalization
    ↓
X_features (57 features) + y_time + y_event
    ↓
Survival Model (Cox PH, Stepwise, LASSO, etc.)
    ↓
Trained Model (coefficients, baseline hazard)
```

### Prediction Phase
```
New Patient Features (X_new) → Trained Model
    ↓
Risk Score → Survival Probability S(t|X_new)
    ↓
Risk Stratification (Low/Medium/High)
```

---

## Key Characteristics

1. **Survival Analysis Framework:**
   - Handles right-censored data naturally
   - Models time-to-event directly
   - Accounts for patients who don't experience events

2. **Longitudinal Structure:**
   - Multiple observations per patient
   - Temporal dependencies
   - Patient-level splitting required to avoid data leakage

3. **Class Imbalance:**
   - Very low event rates (0.33-4.69%)
   - Requires appropriate evaluation metrics (C-index, not accuracy)
   - May need class weighting or resampling strategies

4. **Feature Richness:**
   - 57 features including interactions and temporal patterns
   - Both current and historical measurements
   - Captures complex clinical relationships

5. **Clinical Interpretability:**
   - Hazard ratios provide clinical meaning
   - Risk scores enable stratification
   - Survival curves visualize predictions over time

---

## File Locations

### Processed Data
- **3-year dataset:**
  - Features: `time_to_event/data/processed_3year/X_features.csv`
  - Time-to-event: `time_to_event/data/processed_3year/y_time.npy`
  - Event indicator: `time_to_event/data/processed_3year/y_event.npy`
  - Metadata: `time_to_event/data/processed_3year/metadata.csv`

- **10-year dataset:**
  - Features: `time_to_event/data/processed_10year/X_features.csv`
  - Time-to-event: `time_to_event/data/processed_10year/y_time.npy`
  - Event indicator: `time_to_event/data/processed_10year/y_event.npy`
  - Metadata: `time_to_event/data/processed_10year/metadata.csv`

### Analysis Results
- **Reports:** `time_to_event/results/reports/`
- **Figures:** `time_to_event/results/figures/`
- **Comprehensive Report:** `time_to_event/results/COMPREHENSIVE_ANALYSIS_REPORT.md`

---

## Next Steps for ML/DL Models

Based on this input/output specification, the following models are suitable:

1. **DeepSurv:** Neural network extension of Cox PH model
   - Input: 57 features (X)
   - Output: Risk score → Survival probability

2. **DeepHit/DSM:** Deep learning for survival analysis
   - Handles non-proportional hazards
   - Can predict discrete time-to-event

3. **Random Survival Forests:**
   - Tree-based survival model
   - Handles non-linear relationships

4. **Transformer-based Survival Models:**
   - For temporal sequences
   - Can leverage longitudinal structure

**Evaluation Metrics:**
- C-index (concordance)
- Integrated Brier Score (IBS)
- Calibration curves
- Risk stratification performance

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Related Documents:** `COMPREHENSIVE_ANALYSIS_REPORT.md`, `Analysis_conclude.md`
