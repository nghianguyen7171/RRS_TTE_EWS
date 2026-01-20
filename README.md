# Clinical Deterioration Prediction Research Project

A comprehensive research project for predicting clinical deterioration (cardiac arrest/endotracheal intubation) using vital signs and laboratory test results from Chonnam National University Hospital data.

This repository contains two main subprojects:
- **Time-to-Event Analysis**: Survival analysis for predicting when clinical deterioration occurs
- **Early Warning Systems**: Time-series classification for predicting future deterioration events

## Project Overview

This project analyzes two clinical datasets to develop machine learning models for predicting patient deterioration:
- **3-year dataset** (CNUH_3Y.csv): ~317k records from 2017-2019
- **10-year dataset** (10yrs_proc.csv): ~38k records from 2009-2019

### Problem Type
**Binary Classification** - Predict if a patient will experience clinical deterioration (0 = normal, 1 = deterioration)

### Key Characteristics
- Highly imbalanced datasets (0.33% to 4.69% positive class)
- Feature-rich: Vital signs + Lab values + Demographics
- Temporal data available for time-series exploration
- Patient-level data (requires patient-level splitting)

## Project Structure

```
RRS_3y_10y/
├── README.md                    # This file
├── environment.yml              # Conda environment specification
├── requirements.txt             # Python package requirements
├── .gitignore                   # Git ignore rules
├── data/                        # Raw data (not in git, use .keep)
│   ├── .keep
│   ├── CNUH_3Y.csv             # 3-year dataset (large file)
│   └── 10yrs_proc.csv          # 10-year dataset (large file)
├── docs/                        # Documentation
│   ├── dat_info.md             # Data description
│   └── req_test_results.md     # Lab test requirements
├── notebooks/                   # Jupyter notebooks for analysis
│   └── 01_data_overview.ipynb  # Initial data exploration
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── eda_utils.py            # EDA helper functions
│   └── preprocessing.py        # Data preprocessing utilities
├── scripts/                     # Analysis scripts
│   ├── run_eda_analysis.py     # Comprehensive EDA script
│   └── generate_problem_definition.py  # Problem definition analysis
├── reports/                     # Generated reports and visualizations
│   ├── EDA_SUMMARY_REPORT.md   # Summary of EDA findings
│   └── ANALYSIS_RESULTS.md     # Detailed analysis results
├── time_to_event/               # Time-to-event analysis subproject
│   ├── README.md               # Time-to-event specific README
│   ├── R/                      # R analysis scripts
│   ├── src/                    # Python processing code
│   ├── scripts/                # Processing scripts
│   ├── data/                   # Processed data (use .keep)
│   └── results/                # Analysis results
│       ├── COMPREHENSIVE_ANALYSIS_REPORT.md  # Main analysis report
│       ├── figures/            # All analysis figures
│       └── reports/            # Analysis tables and results
└── early_warning/               # Early warning systems subproject
    ├── README.md               # Early warning specific README
    ├── src/                    # Python processing code
    ├── scripts/                # Processing scripts
    ├── data/                   # Processed data (use .keep)
    └── results/                # Analysis results
```

## Installation

### Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate rrs_clinical_deterioration
```

### Using pip

```bash
# Install requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Run EDA Analysis

```bash
# Run comprehensive EDA analysis
cd scripts
python run_eda_analysis.py
```

### 2. Generate Problem Definition

```bash
# Generate problem definition analysis
python generate_problem_definition.py
```

### 3. Run Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_overview.ipynb
# 2. notebooks/02_eda_3year.ipynb
# 3. notebooks/03_eda_10year.ipynb
# 4. notebooks/04_problem_definition.ipynb
```

## Key Findings

### Dataset Summary

| Metric | 3-Year Dataset | 10-Year Dataset |
|--------|---------------|-----------------|
| Total Rows | 317,006 | 37,799 |
| Class 0 (Normal) | 315,964 (99.67%) | 36,026 (95.31%) |
| Class 1 (Deterioration) | 1,042 (0.33%) | 1,773 (4.69%) |
| Imbalance Ratio | 303.2:1 | 20.3:1 |
| Unique Patients | 2,619 | 708 |

### Problem Definition

- **Type:** Binary Classification
- **Challenge:** Extreme class imbalance
- **Approach:** Static classification with patient-level splitting
- **Key Metrics:** AUC-ROC, Recall (critical), Precision, F1-score

### Recommendations

1. **Data Splitting:** Use patient-level stratified split (70/15/15)
2. **Imbalance Handling:** Class weighting, SMOTE, or cost-sensitive learning
3. **Models:** Random Forest, XGBoost, LightGBM
4. **Focus:** High Recall (sensitivity) for clinical safety

## Usage Examples

### Load Datasets

```python
from src.data_loader import load_3year_dataset, load_10year_dataset

# Load datasets
df_3year = load_3year_dataset("data/CNUH_3Y.csv")
df_10year = load_10year_dataset("data/10yrs_proc.csv")
```

### Run EDA

```python
from src.eda_utils import (
    plot_target_distribution,
    plot_feature_distributions,
    calculate_statistics_by_target
)

# Plot target distribution
plot_target_distribution(df_3year, "3-Year Dataset Target Distribution")

# Calculate statistics by target
stats = calculate_statistics_by_target(df_3year, ['HR', 'SBP', 'SaO2'], 'target')
```

## Results

### Time-to-Event Analysis

Comprehensive survival analysis results are available in:
- **Main Report**: `time_to_event/results/COMPREHENSIVE_ANALYSIS_REPORT.md`
- **Figures**: `time_to_event/results/figures/`
- **Tables**: `time_to_event/results/reports/`

Key findings:
- Multivariate Cox models achieved C-index of 0.885 (3-year) and 0.778 (10-year)
- Top predictors: CRP, Albumin, vital sign variability
- Risk stratification successfully identifies high-risk patients

### Early Warning Systems

Early warning analysis results are available in:
- `early_warning/results/`

### Initial EDA

See `reports/EDA_SUMMARY_REPORT.md` for comprehensive EDA findings.

## Data Description

See `docs/dat_info.md` for detailed information about the datasets.

## Contributing

This is a research project. For questions or contributions, please refer to the project documentation.

## License

[Specify license if applicable]

## Citation

If you use this project in your research, please cite appropriately.

## Contact

[Add contact information if needed]
