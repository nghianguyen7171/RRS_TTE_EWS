# Load and prepare survival analysis data
# This script loads processed data from Python preprocessing pipeline

# Load required libraries
library(reticulate)
library(dplyr)
library(readr)

# Source utility functions
source("R/utils.R")

# Set project directory
project_root <- get_project_root()
setwd(project_root)

# Function to load dataset
load_survival_data <- function(dataset = c("3year", "10year")) {
  dataset <- match.arg(dataset)
  cat("Loading", dataset, "dataset...\n")
  
  data_dir <- get_data_dir(dataset)
  
  # Load metadata
  metadata_path <- file.path(data_dir, "metadata.csv")
  if (!file.exists(metadata_path)) {
    stop("Metadata file not found: ", metadata_path)
  }
  metadata <- read_csv(metadata_path, show_col_types = FALSE)
  cat("Loaded metadata:", nrow(metadata), "rows\n")
  
  # Load features
  features_path <- file.path(data_dir, "X_features.csv")
  if (!file.exists(features_path)) {
    stop("Features file not found: ", features_path)
  }
  
  # Read features in chunks if file is large
  cat("Loading features...\n")
  features <- read_csv(features_path, show_col_types = FALSE)
  cat("Loaded features:", nrow(features), "rows,", ncol(features), "columns\n")
  
  # Load numpy arrays (time-to-event and event indicator)
  y_time_path <- file.path(data_dir, "y_time.npy")
  y_event_path <- file.path(data_dir, "y_event.npy")
  
  if (!file.exists(y_time_path) || !file.exists(y_event_path)) {
    stop("Numpy array files not found")
  }
  
  cat("Loading numpy arrays...\n")
  y_time <- load_numpy_array(y_time_path)
  y_event <- load_numpy_array(y_event_path)
  
  cat("Loaded y_time:", length(y_time), "values\n")
  cat("Loaded y_event:", length(y_event), "values\n")
  
  # Validate dimensions
  if (nrow(features) != length(y_time) || nrow(features) != length(y_event)) {
    stop("Dimension mismatch: features (", nrow(features), 
         ") vs y_time (", length(y_time), ") vs y_event (", length(y_event), ")")
  }
  
  if (nrow(metadata) != nrow(features)) {
    stop("Dimension mismatch: metadata (", nrow(metadata), 
         ") vs features (", nrow(features), ")")
  }
  
  # Check if metadata already has time_to_event and event_occurred
  # If so, use those; otherwise use numpy arrays
  if ("time_to_event" %in% colnames(metadata) && "event_occurred" %in% colnames(metadata)) {
    cat("Using time_to_event and event_occurred from metadata\n")
    # Remove them from metadata temporarily to avoid duplicates
    metadata_clean <- metadata %>% dplyr::select(-time_to_event, -event_occurred)
    survival_data <- cbind(
      metadata_clean,
      features,
      data.frame(
        time_to_event = metadata$time_to_event,
        event_occurred = metadata$event_occurred
      )
    )
  } else {
    # Use numpy arrays
    cat("Using time_to_event and event_occurred from numpy arrays\n")
    survival_data <- cbind(
      metadata,
      features,
      data.frame(
        time_to_event = y_time,
        event_occurred = y_event
      )
    )
  }
  
  # Create survival object
  library(survival)
  survival_data$surv_obj <- Surv(
    time = survival_data$time_to_event,
    event = survival_data$event_occurred
  )
  
  # Basic summary
  cat("\n=== Dataset Summary ===\n")
  cat("Total observations:", nrow(survival_data), "\n")
  cat("Total features:", ncol(features), "\n")
  cat("Events:", sum(survival_data$event_occurred), 
      "(", round(100 * mean(survival_data$event_occurred), 2), "%)\n")
  cat("Censored:", sum(1 - survival_data$event_occurred), 
      "(", round(100 * (1 - mean(survival_data$event_occurred)), 2), "%)\n")
  cat("Unique patients:", length(unique(survival_data$patient_id)), "\n")
  cat("Median time-to-event (events):", 
      median(survival_data$time_to_event[survival_data$event_occurred == 1], na.rm = TRUE), "hours\n")
  cat("Median time-to-event (censored):", 
      median(survival_data$time_to_event[survival_data$event_occurred == 0], na.rm = TRUE), "hours\n")
  
  return(list(
    data = survival_data,
    features = features,
    metadata = metadata,
    feature_names = colnames(features),
    dataset_name = dataset
  ))
}

# Load both datasets
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Loading Survival Analysis Data\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

data_3year <- load_survival_data("3year")
data_10year <- load_survival_data("10year")

cat("\nData loading complete!\n")
cat("Use data_3year and data_10year objects for analysis.\n")
