# R-Based Survival Analysis

This directory contains R scripts for comprehensive survival analysis of the time-to-event data.

## Scripts Overview

### 1. `01_load_data.R`
- Loads processed data from Python preprocessing pipeline
- Handles both 3-year and 10-year datasets
- Creates survival objects (Surv(time, event))
- Validates data dimensions and structure

### 2. `02_descriptive_analysis.R`
- Summary statistics for time-to-event
- Event rate analysis
- Censoring patterns
- Feature distributions by event status
- Patient-level summaries
- Generates descriptive visualizations

### 3. `03_kaplan_meier.R`
- Overall survival curves
- Stratified survival curves by key features
- Log-rank tests for group comparisons
- Median survival times
- Survival probabilities at key time points

### 4. `04_cox_ph.R`
- Univariate Cox proportional hazards models
- Hazard ratios and confidence intervals
- Forest plots
- Basic multivariate Cox models

### 5. `05_multivariate_analysis.R`
- Stepwise feature selection
- LASSO regularization for feature selection
- Model comparison (AIC, BIC)
- Interaction term analysis

### 6. `06_feature_importance.R`
- Feature importance rankings
- Risk score development
- Survival analysis by risk groups

### 7. `07_model_diagnostics.R`
- Proportional hazards assumption testing
- Residual analysis (Schoenfeld, Martingale, Deviance)
- Model fit assessment

### 8. `08_comparison.R`
- Comparison of 3-year vs 10-year datasets
- Model performance comparison
- Feature consistency analysis
- Survival curve comparisons

### 9. `utils.R`
- Utility functions for data loading, figure saving, etc.
- Helper functions for formatting and calculations

### 10. `run_all_analyses.R`
- Master script to run all analyses in sequence

## Usage

### Run All Analyses
```r
# From the time_to_event directory
setwd("time_to_event")
source("R/run_all_analyses.R")
```

### Run Individual Analyses
```r
# Load data first
source("R/01_load_data.R")

# Then run specific analyses
source("R/02_descriptive_analysis.R")
source("R/03_kaplan_meier.R")
# etc.
```

## Required R Packages

Install required packages:
```r
install.packages(c(
  "survival",
  "survminer",
  "ggplot2",
  "dplyr",
  "tidyr",
  "reticulate",
  "forestplot",
  "glmnet",
  "rms",
  "survivalROC",
  "MASS"
))
```

### Loading NumPy Arrays

The scripts use `reticulate` to load numpy arrays. Make sure Python and numpy are available:
```r
library(reticulate)
py_install("numpy")
```

## Output Structure

Results are saved to:
- `results/figures/` - All plots and visualizations
  - `km_curves/` - Kaplan-Meier survival curves
  - `cox_models/` - Cox model visualizations
  - `diagnostics/` - Model diagnostic plots
  - `comparisons/` - Dataset comparison plots
- `results/reports/` - Analysis results and tables
  - `descriptive_stats/` - Descriptive statistics
  - `cox_results/` - Cox model results
  - `model_summaries/` - Model summaries and comparisons

## Notes

- All scripts assume data has been preprocessed using the Python pipeline
- Missing values are handled by removing rows with any missing data
- Large datasets may require significant memory and processing time
- Some analyses (e.g., univariate Cox for all features) may take a while to complete
