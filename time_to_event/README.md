# Time-to-Event Analysis

This subproject contains comprehensive survival analysis for predicting when clinical deterioration events occur. It includes both Python data processing pipelines and R-based statistical analysis.

## Overview

The time-to-event models predict **when** a clinical deterioration event will occur. This is a survival analysis problem where:
- **Target**: Time until event (continuous, in hours)
- **Event indicator**: Whether event occurred (binary: 0=censored, 1=event)
- **Censoring**: Patients without events are right-censored

## Project Structure

```
time_to_event/
├── README.md               # This file
├── data/                   # Processed datasets (large files, use .keep)
│   ├── processed_3year/   # 3-year processed data
│   │   ├── .keep
│   │   ├── X_features.csv (large)
│   │   ├── y_time.npy (large)
│   │   ├── y_event.npy (large)
│   │   └── metadata.csv
│   └── processed_10year/  # 10-year processed data
│       ├── .keep
│       ├── X_features.csv (large)
│       ├── y_time.npy (large)
│       ├── y_event.npy (large)
│       └── metadata.csv
├── src/                    # Python processing code
│   ├── data_loader.py     # Load and validate datasets
│   ├── preprocessing.py   # Data cleaning and preparation
│   ├── survival_labels.py # Create time-to-event labels
│   ├── feature_engineering.py  # Feature creation
│   └── validation.py      # Data validation
├── scripts/                # Processing scripts
│   ├── process_3year_dataset.py
│   ├── process_10year_dataset.py
│   └── validate_processed_data.py
├── R/                      # R analysis scripts
│   ├── 01_load_data.R     # Data loading
│   ├── 02_descriptive_analysis.R
│   ├── 03_kaplan_meier.R  # Kaplan-Meier analysis
│   ├── 04_cox_ph.R       # Cox PH models
│   ├── 05_multivariate_analysis.R
│   ├── 06_feature_importance.R
│   ├── 07_model_diagnostics.R
│   ├── 08_comparison.R   # Dataset comparison
│   ├── utils.R           # Utility functions
│   └── run_all_analyses.R # Master script
├── notebooks/             # Analysis notebooks
└── results/               # Analysis results
    ├── COMPREHENSIVE_ANALYSIS_REPORT.md  # Main report with figures
    ├── figures/           # All analysis figures
    │   ├── km_curves/    # Kaplan-Meier curves
    │   ├── cox_models/   # Cox model visualizations
    │   ├── diagnostics/  # Model diagnostics
    │   └── comparisons/  # Dataset comparisons
    └── reports/          # Analysis tables
        ├── descriptive_stats/
        ├── cox_results/
        └── model_summaries/
```

## Data Processing Pipeline

1. **Load Data**: Load raw datasets and impute missing event times
2. **Preprocess**: Handle missing values, outliers, normalize features
3. **Feature Engineering**: Create temporal and interaction features
4. **Survival Labels**: Calculate time-to-event and handle censoring
5. **Validation**: Validate processed data quality
6. **Save**: Save model-ready datasets

## Output Format

Processed data includes:
- **X_features.csv**: Feature matrix (n_samples × n_features)
- **y_time.npy**: Time-to-event values (continuous, hours)
- **y_event.npy**: Event indicator (binary: 0=censored, 1=event)
- **metadata.csv**: Patient IDs, measurement times, event times

## Usage

### Process 3-year dataset:
```bash
cd time_to_event/scripts
python process_3year_dataset.py
```

### Process 10-year dataset:
```bash
cd time_to_event/scripts
python process_10year_dataset.py
```

### Validate processed data:
```bash
cd time_to_event/scripts
python validate_processed_data.py
```

## Key Features

- **Event time imputation**: Automatically imputes missing event times from target column
- **Censoring handling**: Properly handles right-censored data (patients without events)
- **Temporal features**: Creates time-based features (time since admission, time of day)
- **Feature engineering**: Interaction features and aggregated statistics
- **Data validation**: Comprehensive validation of processed data

## R-Based Survival Analysis

After data processing, comprehensive survival analysis was performed using R:

### Running the Analysis

```bash
cd time_to_event
Rscript R/run_all_analyses.R
```

Or run individual analyses:
```r
source("R/01_load_data.R")
source("R/02_descriptive_analysis.R")
source("R/03_kaplan_meier.R")
# etc.
```

### Analysis Components

1. **Descriptive Analysis**: Summary statistics, event rates, censoring patterns
2. **Kaplan-Meier Analysis**: Survival curves, log-rank tests, risk stratification
3. **Cox PH Models**: Univariate and multivariate Cox proportional hazards models
4. **Feature Importance**: Rankings and risk score development
5. **Model Diagnostics**: Proportional hazards assumption, residual analysis
6. **Dataset Comparison**: 3-year vs. 10-year comparison

### Results

See `results/COMPREHENSIVE_ANALYSIS_REPORT.md` for the complete analysis report with embedded figures and detailed explanations.

**Key Results:**
- Multivariate Cox models: C-index 0.885 (3-year), 0.778 (10-year)
- Top predictors: CRP, Albumin, SBP variability, vital sign interactions
- Risk stratification: 4-tier risk groups with clear survival differences

## Model-Ready Data

After processing, the data is ready for:
- Cox Proportional Hazards models
- Random Survival Forests
- Deep Survival Models (DeepSurv, DeepHit)
- Accelerated Failure Time models
