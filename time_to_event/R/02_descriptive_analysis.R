# Descriptive Statistics for Survival Data
# This script performs comprehensive descriptive analysis

# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(survival)

# Source utility functions
source("R/utils.R")

# Set project directory
project_root <- get_project_root()
setwd(project_root)

# Load data (assumes 01_load_data.R has been run)
if (!exists("data_3year") || !exists("data_10year")) {
  source("R/01_load_data.R")
}

# Function to perform descriptive analysis
perform_descriptive_analysis <- function(data_obj, dataset_name) {
  cat("\n=== Descriptive Analysis:", dataset_name, "===\n")
  
  data <- data_obj$data
  
  # 1. Time-to-event summary statistics
  cat("\n--- Time-to-Event Summary ---\n")
  time_summary <- data %>%
    group_by(event_occurred) %>%
    summarise(
      n = n(),
      mean_time = mean(time_to_event, na.rm = TRUE),
      median_time = median(time_to_event, na.rm = TRUE),
      sd_time = sd(time_to_event, na.rm = TRUE),
      min_time = min(time_to_event, na.rm = TRUE),
      max_time = max(time_to_event, na.rm = TRUE),
      q25 = quantile(time_to_event, 0.25, na.rm = TRUE),
      q75 = quantile(time_to_event, 0.75, na.rm = TRUE),
      .groups = "drop"
    )
  
  print(time_summary)
  
  # 2. Event rate analysis
  cat("\n--- Event Rate Analysis ---\n")
  event_rate <- data %>%
    summarise(
      total_obs = n(),
      total_events = sum(event_occurred),
      event_rate = mean(event_occurred),
      censored = sum(1 - event_occurred),
      censoring_rate = 1 - mean(event_occurred)
    )
  
  print(event_rate)
  
  # 3. Patient-level summary
  cat("\n--- Patient-Level Summary ---\n")
  patient_summary <- data %>%
    group_by(patient_id) %>%
    summarise(
      n_measurements = n(),
      has_event = any(event_occurred == 1),
      time_to_event = first(time_to_event[event_occurred == 1], default = first(time_to_event)),
      .groups = "drop"
    ) %>%
    summarise(
      n_patients = n(),
      n_patients_with_event = sum(has_event),
      mean_measurements_per_patient = mean(n_measurements),
      median_measurements_per_patient = median(n_measurements),
      min_measurements = min(n_measurements),
      max_measurements = max(n_measurements)
    )
  
  print(patient_summary)
  
  # 4. Censoring patterns
  cat("\n--- Censoring Patterns ---\n")
  censoring_patterns <- data %>%
    group_by(patient_id) %>%
    summarise(
      event_status = ifelse(any(event_occurred == 1), "Event", "Censored"),
      max_time = max(time_to_event, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(event_status) %>%
    summarise(
      n = n(),
      mean_max_time = mean(max_time, na.rm = TRUE),
      median_max_time = median(max_time, na.rm = TRUE),
      .groups = "drop"
    )
  
  print(censoring_patterns)
  
  # 5. Feature distributions by event status
  cat("\n--- Feature Statistics by Event Status ---\n")
  
  # Select numeric features (exclude metadata)
  numeric_features <- data %>%
    select(-patient_id, -measurement_time, -event_time, -time_to_event, 
           -event_occurred, -surv_obj) %>%
    select_if(is.numeric) %>%
    colnames()
  
  # Limit to first 20 features for summary (to avoid overwhelming output)
  features_to_summarize <- head(numeric_features, 20)
  
  feature_stats <- data %>%
    select(all_of(c("event_occurred", features_to_summarize))) %>%
    group_by(event_occurred) %>%
    summarise(across(everything(), 
                     list(mean = ~mean(.x, na.rm = TRUE),
                          median = ~median(.x, na.rm = TRUE),
                          sd = ~sd(.x, na.rm = TRUE)),
                     .names = "{.col}_{.fn}"),
              .groups = "drop")
  
  # Save summary tables
  save_table(time_summary, 
             paste0("time_summary_", dataset_name, ".csv"),
             "descriptive_stats")
  save_table(event_rate, 
             paste0("event_rate_", dataset_name, ".csv"),
             "descriptive_stats")
  save_table(patient_summary, 
             paste0("patient_summary_", dataset_name, ".csv"),
             "descriptive_stats")
  save_table(censoring_patterns, 
             paste0("censoring_patterns_", dataset_name, ".csv"),
             "descriptive_stats")
  save_table(feature_stats, 
             paste0("feature_stats_", dataset_name, ".csv"),
             "descriptive_stats")
  
  # 6. Visualizations
  
  # Time-to-event distribution
  p1 <- ggplot(data, aes(x = time_to_event, fill = factor(event_occurred))) +
    geom_histogram(alpha = 0.7, bins = 50, position = "identity") +
    facet_wrap(~event_occurred, ncol = 1, 
               labeller = labeller(event_occurred = c("0" = "Censored", "1" = "Event"))) +
    labs(title = paste("Time-to-Event Distribution:", dataset_name),
         x = "Time to Event (hours)",
         y = "Frequency",
         fill = "Status") +
    theme_minimal()
  
  save_figure(p1, paste0("time_distribution_", dataset_name, ".png"), 
              width = 10, height = 8)
  
  # Time-to-event boxplot
  p2 <- ggplot(data, aes(x = factor(event_occurred), y = time_to_event, 
                         fill = factor(event_occurred))) +
    geom_boxplot(alpha = 0.7) +
    scale_x_discrete(labels = c("0" = "Censored", "1" = "Event")) +
    labs(title = paste("Time-to-Event by Event Status:", dataset_name),
         x = "Event Status",
         y = "Time to Event (hours)",
         fill = "Status") +
    theme_minimal()
  
  save_figure(p2, paste0("time_boxplot_", dataset_name, ".png"), 
              width = 8, height = 6)
  
  # Event rate visualization
  p3 <- ggplot(event_rate, aes(x = "Overall", y = event_rate)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_text(aes(label = paste0(round(event_rate * 100, 2), "%")), 
              vjust = -0.5) +
    labs(title = paste("Event Rate:", dataset_name),
         x = "",
         y = "Event Rate") +
    ylim(0, max(event_rate$event_rate) * 1.2) +
    theme_minimal()
  
  save_figure(p3, paste0("event_rate_", dataset_name, ".png"), 
              width = 6, height = 6)
  
  return(list(
    time_summary = time_summary,
    event_rate = event_rate,
    patient_summary = patient_summary,
    censoring_patterns = censoring_patterns,
    feature_stats = feature_stats
  ))
}

# Perform analysis for both datasets
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Descriptive Analysis\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

desc_3year <- perform_descriptive_analysis(data_3year, "3year")
desc_10year <- perform_descriptive_analysis(data_10year, "10year")

cat("\nDescriptive analysis complete!\n")
cat("Results saved to results/reports/descriptive_stats/\n")
cat("Figures saved to results/figures/\n")
