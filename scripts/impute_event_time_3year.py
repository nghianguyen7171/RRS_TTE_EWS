#!/usr/bin/env python3
"""
Impute event_time in the 3-year dataset (CNUH_3Y.csv) from target and measurement_time.

Assumption:
  - For a given patient, the true event time is the last measurement_time where target == 1.

This script:
  1. Loads the 3-year dataset using src.data_loader.load_3year_dataset
  2. Imputes missing event_time values based on (Patient, target, measurement_time)
  3. Validates imputed event times against existing event_time values (where present)
  4. Prints a summary report
  5. Optionally saves an imputed copy of the dataset
"""

import sys
import os
from typing import Dict, Any

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_3year_dataset  # noqa: E402


def impute_event_time_from_target(
    df: pd.DataFrame,
    patient_col: str = "Patient",
    target_col: str = "target",
    measurement_time_col: str = "measurement_time",
    event_time_col: str = "event_time",
) -> pd.DataFrame:
    """
    Impute event_time from target column and measurement_time.

    For patients with target=1, event_time is set to the last measurement_time
    where target=1 for that patient.
    """
    df = df.copy()

    # Ensure datetime
    df[measurement_time_col] = pd.to_datetime(df[measurement_time_col], errors="coerce")
    if event_time_col in df.columns:
        df[event_time_col] = pd.to_datetime(df[event_time_col], errors="coerce")
    else:
        df[event_time_col] = pd.NaT

    positive_mask = df[target_col] == 1
    positive_patients = df.loc[positive_mask, patient_col].dropna().unique()

    print(f"Found {len(positive_patients)} patients with {target_col}=1")

    imputed_rows = 0
    # Precompute for speed: group by patient
    grouped = df.groupby(patient_col)

    for patient_id in positive_patients:
        try:
            patient_data = grouped.get_group(patient_id)
        except KeyError:
            continue

        # Rows where target == 1 for this patient
        positive_rows = patient_data[patient_data[target_col] == 1]
        if positive_rows.empty:
            continue

        # Last measurement_time where target == 1
        last_event_time = positive_rows[measurement_time_col].max()
        if pd.isna(last_event_time):
            continue

        # Impute event_time for this patient where event_time is missing
        mask = (df[patient_col] == patient_id) & (df[event_time_col].isna())
        count_before = imputed_rows
        df.loc[mask, event_time_col] = last_event_time
        imputed_rows += mask.sum()

    print(f"Imputed {imputed_rows} event_time values based on target and measurement_time")
    return df


def validate_imputed_event_times(
    df: pd.DataFrame,
    patient_col: str = "Patient",
    target_col: str = "target",
    measurement_time_col: str = "measurement_time",
    event_time_col: str = "event_time",
) -> Dict[str, Any]:
    """
    Validate imputed event times against existing values (where available).

    For each patient with at least one row where target==1:
      - Compute the "imputed" event time as max(measurement_time where target==1)
      - Compare to existing event_time (if present)
    """
    df = df.copy()
    df[measurement_time_col] = pd.to_datetime(df[measurement_time_col], errors="coerce")
    df[event_time_col] = pd.to_datetime(df[event_time_col], errors="coerce")

    positive_mask = df[target_col] == 1
    positive_patients = df.loc[positive_mask, patient_col].dropna().unique()

    stats: Dict[str, Any] = {
        "patients_with_target_1": int(len(positive_patients)),
        "patients_with_existing_event_time": 0,
        "patients_with_imputed_event_time": 0,
        "existing_vs_imputed_matches": 0,
        "existing_vs_imputed_differences": 0,
        "max_time_difference_hours": 0.0,
    }

    grouped = df.groupby(patient_col)

    for patient_id in positive_patients:
        try:
            patient_data = grouped.get_group(patient_id)
        except KeyError:
            continue

        positive_rows = patient_data[patient_data[target_col] == 1]
        if positive_rows.empty:
            continue

        # Imputed event time from measurement_time
        imputed_event_time = positive_rows[measurement_time_col].max()
        if pd.isna(imputed_event_time):
            continue

        stats["patients_with_imputed_event_time"] += 1

        existing_event_times = patient_data[event_time_col].dropna().unique()
        if len(existing_event_times) == 0:
            continue

        stats["patients_with_existing_event_time"] += 1

        # Assume a single true event_time per patient if multiple (take first)
        existing_event_time = pd.to_datetime(existing_event_times[0])
        time_diff_hours = abs(
            (imputed_event_time - existing_event_time).total_seconds()
        ) / 3600.0

        if time_diff_hours < 1.0:
            stats["existing_vs_imputed_matches"] += 1
        else:
            stats["existing_vs_imputed_differences"] += 1
            if time_diff_hours > stats["max_time_difference_hours"]:
                stats["max_time_difference_hours"] = float(time_diff_hours)

    return stats


def main() -> None:
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "CNUH_3Y.csv")
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "CNUH_3Y_imputed.csv"
    )

    print("=" * 80)
    print("IMPUTE EVENT_TIME FOR 3-YEAR DATASET (CNUH_3Y.csv)")
    print("=" * 80)

    # Load data
    df = load_3year_dataset(data_path)

    print(f"\nLoaded dataset: {data_path}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Basic counts before
    total_rows = len(df)
    positive_rows = df[df["target"] == 1]
    num_positive = len(positive_rows)
    existing_event_times = df["event_time"].notna().sum()

    print("\nBEFORE IMPUTATION")
    print("-" * 80)
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with target=1: {num_positive:,}")
    print(
        f"Rows with non-null event_time: {existing_event_times:,} "
        f"({existing_event_times / total_rows * 100:.2f}%)"
    )

    # Impute
    df_imputed = impute_event_time_from_target(df)

    # Counts after
    total_event_times_after = df_imputed["event_time"].notna().sum()
    positive_with_event_time = df_imputed.loc[
        df_imputed["target"] == 1, "event_time"
    ].notna().sum()

    print("\nAFTER IMPUTATION")
    print("-" * 80)
    print(
        f"Rows with non-null event_time: {total_event_times_after:,} "
        f"({total_event_times_after / total_rows * 100:.2f}%)"
    )
    print(
        f"Positive rows (target=1) with non-null event_time: "
        f"{positive_with_event_time:,} "
        f"({positive_with_event_time / num_positive * 100:.2f}%)"
    )

    # Validation (using original df with original event_time)
    print("\nVALIDATION AGAINST EXISTING EVENT_TIME")
    print("-" * 80)
    stats = validate_imputed_event_times(df, event_time_col="event_time")

    print(f"Patients with target=1: {stats['patients_with_target_1']:,}")
    print(
        f"Patients with any existing event_time: "
        f"{stats['patients_with_existing_event_time']:,}"
    )
    print(
        f"Patients with imputed event_time (from target/measurement_time): "
        f"{stats['patients_with_imputed_event_time']:,}"
    )
    print(
        f"Existing vs imputed matches (<1 hour diff): "
        f"{stats['existing_vs_imputed_matches']:,}"
    )
    print(
        f"Existing vs imputed differences (>=1 hour diff): "
        f"{stats['existing_vs_imputed_differences']:,}"
    )
    print(
        f"Maximum time difference between existing and imputed event_time: "
        f"{stats['max_time_difference_hours']:.2f} hours"
    )

    # Save imputed dataset
    print("\nSAVING IMPUTED DATASET")
    print("-" * 80)
    print(f"Saving imputed dataset to: {output_path}")
    df_imputed.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

