# Early Warning Data Processing

This subproject contains data processing pipelines for early warning (temporal classification) models.

## Overview

The early warning models predict **if** a clinical deterioration event will occur within a specific time window. This is a temporal classification problem where:
- **Input**: Sequence of patient measurements over time
- **Output**: Binary classification for each prediction window (6h, 12h, 24h, 48h)
- **Task**: Predict "Will event occur in next X hours?"

## Project Structure

```
early_warning/
├── data/                    # Processed datasets
│   ├── processed_3year/    # 3-year processed sequences
│   └── processed_10year/   # 10-year processed sequences
├── src/                     # Processing code
│   ├── data_loader.py      # Load and validate datasets
│   ├── preprocessing.py    # Data cleaning and preparation
│   ├── sequence_creator.py # Create temporal sequences
│   ├── label_generator.py  # Generate prediction window labels
│   └── validation.py       # Data validation
├── scripts/                 # Processing scripts
│   ├── create_sequences_3year.py
│   ├── create_sequences_10year.py
│   └── validate_sequences.py
├── notebooks/              # Analysis notebooks
└── results/                # Processing reports and summaries
```

## Data Processing Pipeline

1. **Load Data**: Load raw datasets and impute missing event times
2. **Preprocess**: Handle missing values, outliers, normalize features
3. **Create Sequences**: Generate sliding window sequences from patient data
4. **Generate Labels**: Create binary labels for each prediction window
5. **Pad Sequences**: Handle variable-length sequences
6. **Validation**: Validate sequence structure and labels
7. **Save**: Save model-ready sequences and labels

## Output Format

Processed data includes:
- **X_sequences.npy**: Padded sequence array (n_sequences × sequence_length × n_features)
- **sequence_lengths.npy**: Actual length of each sequence
- **event_in_6h.npy**: Binary labels for 6-hour window
- **event_in_12h.npy**: Binary labels for 12-hour window
- **event_in_24h.npy**: Binary labels for 24-hour window
- **event_in_48h.npy**: Binary labels for 48-hour window
- **metadata.csv**: Patient IDs, sequence times, etc.

## Usage

### Create sequences from 3-year dataset:
```bash
cd early_warning/scripts
python create_sequences_3year.py
```

### Create sequences from 10-year dataset:
```bash
cd early_warning/scripts
python create_sequences_10year.py
```

### Validate sequences:
```bash
cd early_warning/scripts
python validate_sequences.py
```

## Key Features

- **Sliding window sequences**: Creates multiple sequences per patient using sliding windows
- **Multiple prediction windows**: Generates labels for 6h, 12h, 24h, and 48h windows
- **Variable length handling**: Pads sequences to fixed length while preserving actual lengths
- **Feature normalization**: Standardizes features across sequences
- **Data validation**: Comprehensive validation of sequence structure and labels

## Model-Ready Data

After processing, the data is ready for:
- LSTM/GRU models
- Transformer models
- Time-series CNNs
- Attention-based models

## Sequence Parameters

Default parameters:
- **Sequence length**: 10 measurements
- **Stride**: 1 (every measurement)
- **Min sequence length**: 5 measurements
- **Prediction windows**: [6, 12, 24, 48] hours

These can be adjusted in the processing scripts.
